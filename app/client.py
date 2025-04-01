"""
This script creates an OpenAI Request demo for the CogAgent model.
All parameters are now controlled by arguments, allowing you to run the script
with a single line of code specifying all needed parameters.
"""

import argparse
import base64
import platform
import pyautogui
import re
import os
import gradio as gr
import threading
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
# Import the `agent` function from `register` (assumed to be in the same directory)
from register import agent

stop_event = threading.Event()


def create_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_length: int = 512,
    top_p: float = 1.0,
    temperature: float = 1.0,
    presence_penalty: float = 1.0,
    stream: bool = False,
) -> Any:
    """
    Creates a chat completion request.

    Parameters:
    - api_key (str): API key for authentication.
    - base_url (str): The base URL for the API endpoint.
    - model (str): Model name to use (e.g., "cogagent-9b-20241220").
    - messages (List[Dict[str, Any]]): A list of messages for the conversation, where each message is a dictionary.
    - max_length (int, optional): The maximum length of the response. Default is 512.
    - top_p (float, optional): Controls nucleus sampling. Default is 1.0.
    - temperature (float, optional): Sampling temperature to control randomness. Default is 1.0.
    - presence_penalty (float, optional): Presence penalty for the model. Default is 1.0.
    - stream (bool, optional): Whether to stream the response. Default is False.

    Returns:
    - Any: The response from the chat completion API.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        timeout=60,
        max_tokens=max_length,
        temperature=temperature,
        presence_penalty=presence_penalty,
        top_p=top_p,
    )
    if response:
            return response.choices[0].message.content

def encode_image(image_path: str) -> str:
    """
    Encodes an image file into a base64 string.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        str: The base64-encoded string representation of the image.

    Raises:
        FileNotFoundError: If the specified image file is not found.
        IOError: If an error occurs during file reading.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {image_path}")
    except IOError as e:
        raise IOError(f"Error reading file {image_path}: {e}")


def identify_os() -> str:
    """
    Identifies the operating system based on the platform information.

    Returns:
    - str: "Mac" if the system is macOS, "WIN" if the system is Windows.

    Raises:
    - ValueError: If the operating system is not supported.
    """

    #TODO: Need check if windows platform can run the demo.

    os_detail = platform.platform().lower()
    if "mac" in os_detail:
        return "Mac"
    elif "windows" in os_detail:
        return "WIN"
    else:
        raise ValueError(
            f"This {os_detail} operating system is not currently supported!"
        )


def formatting_input(
    task: str, history_step: List[str], history_action: List[str], round_num: int
) -> List[Dict[str, Any]]:
    """
    Formats input data into a structured message for further processing.

    Parameters:
    - task (str): The task or query the user is asking about.
    - history_step (List[str]): A list of historical steps in the conversation.
    - history_action (List[str]): A list of actions corresponding to the history steps.
    - round_num (int): The current round number (used to identify the image file).

    Returns:
    - List[Dict[str, Any]]: A list of messages formatted as dictionaries.

    Raises:
    - ValueError: If the lengths of `history_step` and `history_action` do not match.
    """
    current_platform = identify_os()
    platform_str = f"(Platform: {current_platform})\n"
    format_str = "(Answer in Status-Plan-Action-Operation-Sensitive format.)\n"

    if len(history_step) != len(history_action):
        raise ValueError("Mismatch in lengths of history_step and history_action.")

    history_str = "\nHistory steps: "
    for index, (step, action) in enumerate(zip(history_step, history_action)):
        history_str += f"\n{index}. {step}\t{action}"

    query = f"Task: {task}{history_str}\n{platform_str}{format_str}"

    # Create image URL with base64 encoding
    img_url = f"data:image/jpeg;base64,{encode_image(f'caches/img_{round_num}.png')}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_url},
                },
            ],
        },
    ]
    return messages


def shot_current_screen(round_num: int):
    """
    Captures a screenshot of the current screen and saves it to the cache directory.

    Parameters:
    - round_num (int): The current round number for naming the image file.
    """
    img = pyautogui.screenshot()
    img.save(f"caches/img_{round_num}.png")


def extract_grounded_operation(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the grounded operation and action from the response text.

    Parameters:
    - response (str): The model's response text.

    Returns:
    - (step, action) (Tuple[Optional[str], Optional[str]]): Extracted step and action from the response.
    """
    grounded_pattern = r"Grounded Operation:\s*(.*)"
    action_pattern = r"Action:\s*(.*)"

    step = None
    action = None

    matches_history = re.search(grounded_pattern, response)
    matches_actions = re.search(action_pattern, response)
    if matches_history:
        step = matches_history.group(1)
    if matches_actions:
        action = matches_actions.group(1)

    return step, action


def draw_boxes_on_image(image: Image.Image, boxes: List[List[float]], save_path: str):
    """
    Draws red bounding boxes on the given image and saves it.

    Parameters:
    - image (PIL.Image.Image): The image on which to draw the bounding boxes.
    - boxes (List[List[float]]): A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].
      Coordinates are expected to be normalized (0 to 1).
    - save_path (str): The path to save the updated image.
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min = int(box[0] * image.width)
        y_min = int(box[1] * image.height)
        x_max = int(box[2] * image.width)
        y_max = int(box[3] * image.height)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    image.save(save_path)


def extract_bboxes(response: str, round_num: int):
    """
    Extracts bounding boxes from the response and draws them on the corresponding screenshot.

    Parameters:
    - response (str): The response text containing bounding box information.
    - round_num (int): The round number to identify which image to annotate.
    """
    box_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
    matches = re.findall(box_pattern, response)
    if matches:
        boxes = [[int(x) / 1000 for x in match] for match in matches]
        img_save_path = f"caches/img_{round_num}_bbox.png"
        image = Image.open(f"caches/img_{round_num}.png").convert("RGB")
        draw_boxes_on_image(image, boxes, img_save_path)


def is_balanced(s: str) -> bool:
    """
    Checks if the parentheses in a string are balanced.

    Parameters:
    - s (str): The string to check.

    Returns:
    - bool: True if parentheses are balanced, False otherwise.
    """
    stack = []
    mapping = {")": "(", "]": "[", "}": "{"}
    if "(" not in s:
        return False
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or mapping[char] != stack.pop():
                return False
    return not stack


def extract_operation(step: Optional[str]) -> Dict[str, Any]:
    """
    Extracts the operation and other details from the grounded operation step.

    Parameters:
    - step (Optional[str]): The grounded operation string.

    Returns:
    - Dict[str, Any]: A dictionary containing the operation details.
    """
    if step is None or not is_balanced(step):
        return {"operation": "NO_ACTION"}

    op, detail = step.split("(", 1)
    detail = "(" + detail
    others_pattern = r"(\w+)\s*=\s*([^,)]+)"
    others = re.findall(others_pattern, detail)
    Grounded_Operation = dict(others)

    boxes_pattern = r"box=\[\[(.*?)\]\]"
    boxes = re.findall(boxes_pattern, detail)
    if boxes:
        Grounded_Operation["box"] = list(map(int, boxes[0].split(",")))
    Grounded_Operation["operation"] = op.strip()

    return Grounded_Operation


def workflow(
    api_key: str,
    base_url: str,
    model: str,
    chatbot: List[List[str]],
    max_length: int,
    top_p: float,
    temperature: float,
):
    """
    Main workflow for handling a chatbot interaction loop.

    Parameters:
    - api_key (str): API key for accessing the chatbot API.
    - base_url (str): Base URL for the chatbot API.
    - model (str): Model name to use.
    - chatbot (list): The initial history of the chatbot interaction.
    - max_length (int): Maximum response length for the chatbot.
    - top_p (float): Top-p sampling value for response generation.
    - temperature (float): Temperature value for response randomness.

    Yields:
    - history (list): Updated history of the chatbot interaction.
    - output_image (str): Path to the generated output image.
    """
    history_step = []
    history_action = []
    history = chatbot
    round_num = 1
    task = chatbot[-1][0] if chatbot and chatbot[-1] else "No task provided"

    try:
        while True:
            print(f"\033[92m Round {round_num}: \033[0m")
            if round_num > 15:
                break  # Exit the loop after 15 rounds

            # Capture the current screen for the round
            shot_current_screen(round_num)

            # Format input messages for the chatbot
            messages = formatting_input(task, history_step, history_action, round_num)

            # Call the chatbot API to get a response
            response = create_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                stream=False,
            )

            # Extract grounded operations and actions from the response
            step, action = extract_grounded_operation(response)
            history_step.append(step if step else "")
            history_action.append(action if action else "")

            # Process bounding boxes and operations
            extract_bboxes(response, round_num)
            grounded_operation = extract_operation(step)

            if grounded_operation["operation"] == "NO_ACTION":
                break

            # Execute the grounded operation using the agent
            status = agent(grounded_operation)

            # Update the history with the latest response
            history.append([f"Round {round_num}", response])

            # Prepare the output image path
            output_image = f"caches/img_{round_num}_bbox.png"
            if status == "END" or stop_event.is_set():
                output_image = f"caches/img_{round_num - 1}_bbox.png"
                yield history, output_image
                break
            else:
                yield history, output_image

            round_num += 1
    finally:
        # Clear the stop event at the end of the workflow
        stop_event.clear()


def switch():
    """
    Sets the stop event to terminate the workflow.
    """
    stop_event.set()


def gradio_web(
    api_key: str,
    base_url: str,
    model: str,
    client_name: str = "127.0.0.1",
    client_port: int = 8080,
):
    """
    Launches a Gradio-based web application for interacting with CogAgent.

    Parameters:
    - api_key (str): OpenAI API key.
    - base_url (str): OpenAI API base URL.
    - model (str): Model name to use.
    - presence_penalty (float): Presence penalty for the model.
    - client_name (str): The gradio IP or hostname for hosting the app.
    - client_port (int): The port number for the gradio.
    """
    with gr.Blocks() as demo:
        gr.HTML("<h1 align='center'>CogAgent Gradio Chat Demo</h1>")

        # Top row: Chatbot and Image upload
        with gr.Row():
            with gr.Column(scale=1, min_width=160):
                chatbot = gr.Chatbot(height=240)
            with gr.Column(scale=1, min_width=160):
                img_path = gr.Image(
                    label="Operation Area Screenshot",
                    type="filepath",
                    show_fullscreen_button=True,
                )

        # Bottom row: Task input, system controls
        with gr.Row():
            with gr.Column(scale=1, min_width=160):
                task = gr.Textbox(
                    show_label=True,
                    placeholder="Please enter your task description",
                    label="Task",
                    max_length=320,
                )
                submit_button = gr.Button("Submit")
                clear_button = gr.Button("Clear History")

            with gr.Column(scale=1, min_width=160):
                max_length = gr.Slider(
                    minimum=0,
                    maximum=8192,
                    value=4096,
                    step=1.0,
                    label="Maximum Length",
                    interactive=True,
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                    label="Top P",
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.6,
                    step=0.01,
                    label="Temperature",
                    interactive=True,
                )

        # Stop button
        with gr.Row():
            with gr.Column(scale=1, min_width=160):
                stop_button = gr.Button("Stop", variant="stop", size="lg")

        # Define functions for button actions
        def user_input(task, history):
            """Handles user task submission."""
            return "", history + [[task, "Please wait for CogAgent's operation..."]]

        def raise_error():
            """Raises an error to terminate the program."""
            raise gr.Error("The program has been terminated!")

        def warning_start():
            """Displays a warning when CogAgent starts processing."""
            return gr.Warning(
                "CogAgent is processing. Please do not interact with the keyboard or mouse.",
                duration=60,
            )

        def warning_end():
            """Displays a warning when CogAgent finishes processing."""
            return gr.Warning("CogAgent has finished. Please input a new task.")

        # Create a partial workflow function with fixed parameters
        workflow_partial = partial(workflow, api_key, base_url, model)

        # Button actions and callbacks
        submit_button.click(
            user_input, inputs=[task, chatbot], outputs=[task, chatbot], queue=False
        ).then(warning_start, inputs=None, outputs=None).then(
            workflow_partial,
            inputs=[chatbot, max_length, top_p, temperature],
            outputs=[chatbot, img_path],
        ).then(warning_end, inputs=None, outputs=None)

        clear_button.click(
            lambda: (None, None), inputs=None, outputs=[chatbot, img_path], queue=False
        )

        stop_button.click(switch).then(raise_error, inputs=None, outputs=None)

    demo.queue()
    demo.launch(server_name=client_name, server_port=client_port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the CogAgent demo with all parameters controlled by command-line arguments."
    )
    parser.add_argument("--api_key", required=True, help="OpenAI API Key.")
    parser.add_argument("--base_url", required=True, help="OpenAI API Base URL.")
    parser.add_argument("--model", default="CogAgent", help="Model name to use.")
    parser.add_argument(
        "--client_name",
        default="127.0.0.1",
        help="The IP or hostname for Gradio.",
    )
    parser.add_argument(
        "--client_port", type=int, default=8080, help="The port number for Gradio."
    )

    args = parser.parse_args()

    if not os.path.exists("caches"):
        os.makedirs("caches")

    gradio_web(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        client_name=args.client_name,
        client_port=args.client_port,
    )
