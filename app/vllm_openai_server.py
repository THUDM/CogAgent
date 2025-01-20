"""
This script creates a vLLM OpenAI Server demo with vLLM for the CogAgent model,
using the OpenAI API to interact with the model.

You can specify the model path, host, and port via command-line arguments, for example:
python vllm_openai_demo.py --model_path THUDM/cogagent-9b-20241220 --host 0.0.0.0 --port 8000
"""

import argparse
import gc
import time
import base64
from contextlib import asynccontextmanager
from typing import List, Literal, Union, Tuple, Optional
import torch
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from PIL import Image
from io import BytesIO

TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential 
    for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card = ModelCard(id="CogAgent")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    An endpoint to create chat completions given a set of messages and model parameters.
    Returns either a single completion or streams tokens as they are generated.
    """
    global model

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=False,
        repetition_penalty=request.repetition_penalty,
    )

    response = None
    messages = gen_params["messages"]
    query, image = process_history_and_images(messages)

    async for response in vllm_gen(model, query, image):
        pass

    usage = UsageInfo()
    print(response)
    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    
    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage,
    )


def process_history_and_images(
    messages: List[ChatMessageInput],
) -> Tuple[Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    Returns:
        A tuple:
            - query (str): The user's query text.
            - image (PIL.Image.Image or None): The extracted image, if any.
    """
    image = None
    for message in messages:
        content = message.content

        # Extract text content
        if isinstance(content, list):  # text
            text_content = " ".join(
                item.text for item in content if isinstance(item, TextContent)
            )
        else:
            # If content is a string, treat it directly as text
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        # Base64 encoded image
                        base64_encoded_image = image_url.split(
                            "data:image/jpeg;base64,"
                        )[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert("RGB")
                    else:
                        # Fetch image from a remote URL
                        response = requests.get(image_url, verify=False)
                        image = Image.open(BytesIO(response.content)).convert("RGB")

    return text_content, image


async def vllm_gen(
        model: AsyncLLMEngine,
        messages: Optional[str],
        image: Optional[List[Image.Image]]
        ):
    # Use vllm to perform inference. 
    # For details on the meaning of the inputs and params_dict, see vLLM
    inputs = {
        "prompt": messages,
        "multi_modal_data": {"image": image},
    }
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": 0.6,
        "top_p": 0.8,
        "top_k": -1,
        "ignore_eos": False,
        "max_tokens": 8192,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
        "stop_token_ids": [151329, 151336, 151338],
    }
    sampling_params = SamplingParams(**params_dict)

    async for output in model.generate(
        prompt=inputs,
        sampling_params=sampling_params,
        request_id=f"{time.time()}"
    ):
        input_echo_len = len(output.prompt_token_ids) - 1601
        output_echo_len = len(output.outputs[0].token_ids)
        yield {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": output_echo_len,
                "total_tokens": input_echo_len + output_echo_len,
            },
        }


def load_model(model_dir: str):
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=True,
        disable_log_requests=True,
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=8192
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    # Use argparse to control model_path, host, and port from command line arguments
    parser = argparse.ArgumentParser(description="vLLM OpenAI Server Demo for CogAgent")
    parser.add_argument(
        "--model_path", required=True, help="Path or name of the CogAgent model"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port, workers=1)
