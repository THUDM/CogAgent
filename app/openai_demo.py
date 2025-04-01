"""
This script creates an OpenAI Server demo with transformers for the CogAgent model,
using the OpenAI API to interact with the model.

You can specify the model path, host, and port via command-line arguments, for example:
python openai_demo.py --model_path THUDM/cogagent-9b-20241220 --host 0.0.0.0 --port 8000
"""

import argparse
import gc
import threading
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
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from PIL import Image
from io import BytesIO
from pathlib import Path

# Determine the appropriate torch dtype based on the GPU capabilities
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

# Enable CORS so that the API can be called from anywhere
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
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
    )

    if request.stream:
        # If streaming is requested, return an EventSourceResponse that yields tokens as they are generated
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    # Otherwise, return a complete response after generation
    response = generate_cogagent(model, tokenizer, gen_params)

    usage = UsageInfo()
    message = ChatMessageResponse(role="assistant", content=response["text"])
    choice_data = ChatCompletionResponseChoice(index=0, message=message)

    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage,
    )


def predict(model_id: str, params: dict):
    """
    A generator function that streams the model output tokens.
    Used for the `stream=True` scenario, returning tokens as SSE events.
    """
    global model, tokenizer

    # Initially, return the role delta message
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant")
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield chunk.model_dump_json(exclude_unset=True)

    previous_text = ""
    for new_response in generate_stream_cogagent(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text) :]
        previous_text = decoded_unicode
        delta = DeltaMessage(content=delta_text, role="assistant")
        choice_data = ChatCompletionResponseStreamChoice(index=0, delta=delta)
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield chunk.model_dump_json(exclude_unset=True)

    # End of stream message
    choice_data = ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage())
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield chunk.model_dump_json(exclude_unset=True)


def generate_cogagent(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    """
    Generates a response using the CogAgent model.
    It processes the chat history and any provided images,
    and then invokes the model to generate a complete response.
    """
    response = None
    for response in generate_stream_cogagent(model, tokenizer, params):
        pass
    return response


def process_history_and_images(
    messages: List[ChatMessageInput],
) -> Tuple[Optional[str], Optional[Image.Image]]:
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
    text_content = ""
    for message in messages:
        content = message.content

        # Extract text content
        if isinstance(content, list):
            extracted_texts = [
                item.text for item in content if isinstance(item, TextContent)
            ]
            text_content = " ".join(extracted_texts)

        else:
            # If content is a string, treat it directly as text
            text_content = content

        # Extract image content
        if isinstance(content, list):
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


@torch.inference_mode()
def generate_stream_cogagent(model: AutoModel, tokenizer: AutoTokenizer, params: dict):
    """
    Streams the generation results from the model token-by-token.
    Uses TextIteratorStreamer to yield partial responses as they are generated.
    """
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    query, image = process_history_and_images(messages)

    # Apply a chat template (assumed to be provided by the model or custom logic)
    model_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    input_echo_len = len(model_inputs["input_ids"][0])
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = {
        "max_length": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p if temperature > 1e-5 else 0,
        "top_k": 1,
        "streamer": streamer,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    generated_text = ""

    def generate_text():
        with torch.no_grad():
            model.generate(**model_inputs, **gen_kwargs)

    generation_thread = threading.Thread(target=generate_text)
    generation_thread.start()

    total_len = input_echo_len
    for next_text in streamer:
        generated_text += next_text
        total_len = len(tokenizer.encode(generated_text))
        yield {
            "text": generated_text,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }

    generation_thread.join()
    yield {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }


# Clean up GPU memory if possible
gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    # Use argparse to control model_path, host, and port from command line arguments
    parser = argparse.ArgumentParser(description="OpenAI Server Demo for CogAgent")
    parser.add_argument(
        "--model_path", required=True, help="Path or name of the CogAgent model"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path).expanduser().resolve()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, encode_special_tokens=True
    )
    # Load model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    # Run the Uvicorn server with the specified host and port
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
