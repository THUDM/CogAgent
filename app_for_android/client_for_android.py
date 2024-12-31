import re
import os
import base64
import argparse
import platform
import gradio as gr
import time
import threading
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Optional, Tuple
from functools import partial
from openai import OpenAI

from controller import get_screenshot, set_default_input_method
from controller import mobile_agent
from image_compare import wait_screen_loading

####################################### Edit your Setting #########################################
# 1. 确保服务端已经正常运行,并确认服务器端和本地已经通过内网穿透等技术联通
# 2. 请确保服务端可以从外网访问，或者通过内网穿透的方式允许自己的本地访问。
#    在我们的代码中，服务端穿透到本地的端口为`7866`，所以环境变量应该设置为 http://127.0.0.1:7866/v1
API_url = "http://localhost:7866/v1"

# 3. 本 Demo没有设置 API, 因此api_key参数设置为EMPTY。
API_token = "EMPTY"

# Your ADB path
adb_path = ""
###################################################################################################


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
    创建聊天请求。

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
    将图像文件编码为base64字符串。

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


def formatting_input(
    task: str, history_step: List[str], history_action: List[str], round_num: int, image_path: str
) -> List[Dict[str, Any]]:
    """
    Formats input data into a structured message for further processing.

    Parameters:
    - task (str): The task or query the user is asking about.
    - history_step (List[str]): A list of historical steps in the conversation.
    - history_action (List[str]): A list of actions corresponding to the history steps.
    - round_num (int): The current round number (used to identify the image file).
    - image_path (str): 截图保存路径

    Returns:
    - List[Dict[str, Any]]: A list of messages formatted as dictionaries.

    Raises:
    - ValueError: If the lengths of `history_step` and `history_action` do not match.
    """
    current_platform = "Mobile"
    platform_str = f"(Platform: {current_platform})\n"
    format_dict = {
        "action_op_sensitive": "(Answer in Action-Operation-Sensitive format.)\n",
        "status_plan_action_op": "(Answer in Status-Plan-Action-Operation format.)\n",
        "status_action_op_sensitive": "(Answer in Status-Action-Operation-Sensitive format.)\n",
        "status_action_op": "(Answer in Status-Action-Operation format.)\n",
        "action_op": "(Answer in Action-Operation format.)\n",
        "full": "(Answer in Status-Plan-Action-Operation-Sensitive format.)\n",
    }
    format_str = format_dict["full"]

    if len(history_step) != len(history_action):
        raise ValueError("Mismatch in lengths of history_step and history_action.")

    history_str = "\nHistory steps: "
    for index, (step, action) in enumerate(zip(history_step, history_action)):
        history_str += f"\n{index}. {step}\t{action}"

    query = f"Task: {task}{history_str}\n{platform_str}{format_str}"

    # Create image URL with base64 encoding
    img_url = f"data:image/jpeg;base64,{encode_image(image_path)}"

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


def is_balanced(s: str) -> bool:
    """
    Checks if the parentheses in a string are balanced.
    检查字符串中的括号是否平衡。

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


def extract_grounded_operation(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the grounded operation and action from the response text.
    从响应文本中提取固定操作和动作。

    Parameters:
    - response (str): The model's response text. 例如:
        Status: 当前处于手机主屏幕界面[[1, 0, 998, 997]]，饿了么应用图标[[768, 369, 930, 447]]在右下角，已被点击。
        Plan: 1. 打开饿了么应用，进入主界面； 2. 在饿了么应用中找到咖啡瑞幸的选项，点击进入； 3. 在咖啡瑞幸的选项中找到椰子拿铁（冰）的选项，点击进入； 4. 在椰子拿铁（冰）的选项中，选择不另外加糖，并确认； 5. 完成订单后，点击确认，等待配送。
        Action: 打开饿了么应用程序，以便进行后续操作。
        Grounded Operation: LAUNCH(app='饿了么', url='None')

    Returns:
    - (step, action) (Tuple[Optional[str], Optional[str]]): Extracted step and action from the response.
    """
    action_pattern = r"Action:\s*(.*)"
    grounded_pattern = r"Grounded Operation:\s*(.*)"

    step = None
    action = None

    matches_history = re.search(grounded_pattern, response)
    matches_actions = re.search(action_pattern, response)
    if matches_history:
        step = matches_history.group(1)
    if matches_actions:
        action = matches_actions.group(1)

    return step, action


def extract_operation(step: Optional[str]) -> Dict[str, Any]:
    """
    Extracts the operation and other details from the grounded operation step.
    从接地操作步骤中提取操作和其他详细信息。

    Parameters:
    - step (Optional[str]): The grounded operation string. 例如
        Grounded Operation: LAUNCH(app='饿了么', url='None')

    Returns:
    - Dict[str, Any]: A dictionary containing the operation details.
    {
        box: [425, 570, 573, 638],
        element_info: '[android.widget.ImageView]',
        operation: CLICK,
        app: '饿了么',
        url: 'None',
    }
    """
    # 检查括号是否对称
    if step is None or not is_balanced(step):
        return {"operation": "NO_ACTION"}

    op, detail = step.split("(", 1)  # op="LAUNCH"
    detail = "(" + detail  # detail="(app='饿了么', url='None')"
    others_pattern = r"(\w+)\s*=\s*([^,)]+)"
    others = re.findall(others_pattern, detail)
    Grounded_Operation = dict(others)

    boxes_pattern = r"box=\[\[(.*?)\]\]"  # 提取bbox
    boxes = re.findall(boxes_pattern, detail)
    if boxes:
        Grounded_Operation["box"] = list(map(int, boxes[0].split(",")))
    Grounded_Operation["operation"] = op.strip()  # 去除空格

    return Grounded_Operation


def extract_bboxes(image_path: str, response: str, round_num: int):
    """
    Extracts bounding boxes from the response and draws them on the corresponding screenshot.
    从响应中提取边界框，并将其绘制在相应的屏幕截图上。

    Parameters:
    - image_path (str): The path to the image file.
    - response (str): The response text containing bounding box information.
    - round_num (int): The round number to identify which image to annotate.
    """
    box_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
    matches = re.findall(box_pattern, response)
    if matches:
        boxes = [[int(x) / 1000 for x in match] for match in matches]
        image = Image.open(image_path).convert("RGB")
        # img_save_path = image_path + '_bbox'
        img_save_path = f"./caches/img_{round_num}_bbox.png"
        draw_boxes_on_image(image, boxes, img_save_path)

        return img_save_path
    return None


def draw_boxes_on_image(image: Image.Image, boxes: List[List[float]], save_path: str):
    """
    Draws red bounding boxes on the given image and saves it.
    在给定图像上绘制红色边界框并保存。

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
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    image.save(save_path)


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
    处理聊天机器人交互循环的主要工作流程。

    Parameters:
    - api_key (str): API key for accessing the chatbot API.
    - base_url (str): Base URL for the chatbot API.
    - model (str): Model name to use.
    - chatbot (list): The initial history of the chatbot interaction. 聊天机器人交互的最初历史。
    - max_length (int): Maximum response length for the chatbot.
    - top_p (float): Top-p sampling value for response generation.
    - temperature (float): Temperature value for response randomness. 响应随机性的温度值。

    Yields:
    - history (list): Updated history of the chatbot interaction. 聊天机器人交互的更新历史。
    - output_image (str): Path to the generated output image. 生成的输出图像的路径。
    """
    history_step = []
    history_action = []
    history = chatbot
    round_num = 1
    task = chatbot[-1][0] if chatbot and chatbot[-1] else "No task provided"

    try:
        # TODO: 确保设备存在
        set_default_input_method(adb_path)  # 设置默认输入法为 ADB Keyboard
        while round_num < 15:
            # Round i
            print(f"\033[92m Round {round_num}: \033[0m")

            # 获取当前屏幕截图, 保存到"screenshot\screenshot_i.jpg"
            image_path = get_screenshot(adb_path, round_num=round_num)

            # 设置聊天机器人的输入消息格式
            messages = formatting_input(task, history_step, history_action, round_num, image_path)

            # 请求聊天机器人API获取响应
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

            # 从响应中提取 操作和行动
            step, action = extract_grounded_operation(response)
            history_step.append(step if step else "")
            history_action.append(action if action else "")

            # 绘制边框
            output_image = extract_bboxes(image_path, response, round_num)  # 绘制红色边框

            # 提取grounded_operation
            grounded_operation = extract_operation(step)

            # 执行grounded_operation
            status = mobile_agent(grounded_operation, adb_path)
            time.sleep(1)

            # 更新history
            history.append([f"Round {round_num}", response])

            if grounded_operation["operation"] == "NO_ACTION":
                break

            # 输出当前轮次的信息
            if status == "END" or stop_event.is_set():
                output_image = f"caches/img_{round_num - 1}_bbox.png"
                yield history, output_image
                break
            else:
                yield history, output_image

            # 等待执行成功
            print("等待屏幕内容显示完整....")
            wait_screen_loading(adb_path, time_sleep=2, theshold=0.80)

            round_num += 1
    except Exception as e:
        print(f"Error: {e}")
        exit()
    finally:
        # Clear the stop event at the end of the workflow
        # 在工作流结束时清除停止事件
        stop_event.clear()


def gradio_web(
    api_key: str,
    base_url: str,
    model: str,
    client_name: str = "127.0.0.1",
    client_port: int = 8080,
):
    """
    Launches a Gradio-based web application for interacting with CogAgent.
    启动一个基于Gradio的web应用程序，用于与CogAgent交互。

    Parameters:
    - api_key (str): OpenAI API key.
    - base_url (str): OpenAI API base URL.
    - model (str): Model name to use.
    - client_name (str): The gradio IP or hostname for hosting the app.
    - client_port (int): The port number for the gradio.
    """
    with gr.Blocks() as demo:
        gr.HTML("<h1 align='center'>CogAgent Gradio Chat Demo For Android</h1>")

        with gr.Row():
            # Inputs
            with gr.Column(scale=1, min_width=160):
                # Task input
                with gr.Row(min_height=100):
                    task = gr.Textbox(
                        show_label=True,
                        placeholder="Please enter your task description",
                        label="Task",
                        max_length=320,
                    )

                # Model parameters
                with gr.Row(min_height=100):
                    with gr.Column(scale=1, min_width=160):
                        max_length = gr.Slider(
                            minimum=0,
                            maximum=8192,
                            value=4096,
                            step=1.0,
                            label="Maximum Length",
                            interactive=True,
                            min_width=160,
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                            label="Top P",
                            interactive=True,
                            min_width=160,
                        )
                        temperature = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=0.6,
                            step=0.01,
                            label="Temperature",
                            interactive=True,
                            min_width=160,
                        )

                # Stop button
                with gr.Row(min_height=100):
                    submit_button = gr.Button("Submit", variant="primary", size="lg")
                    clear_button = gr.Button("Clear History", size="lg")
                    stop_button = gr.Button("Stop", variant="stop", size="lg")

            # Chatbot
            with gr.Column(scale=1, min_width=160):
                chatbot = gr.Chatbot(height=600, type="tuples")

            # Image input
            with gr.Column(scale=1, min_width=160):
                img_path = gr.Image(
                    label="Operation Area Screenshot",
                    type="filepath",
                    show_fullscreen_button=True,
                    height=600,
                )

        # 设置停止事件以终止工作流。
        def switch():
            """
            Sets the stop event to terminate the workflow.
            设置停止事件以终止工作流。
            """
            stop_event.set()

        # 定义按钮操作的功能
        def user_input(task, history):
            """Handles user task submission."""
            history = history + [[task, "Please wait for CogAgent's operation..."]]
            task = ""
            print(f"{history=}")
            return task, history

        # 引发错误以终止程序。
        def raise_error():
            """Raises an error to terminate the program."""
            raise gr.Error("The program has been terminated!")

        # CogAgent开始处理时显示警告。
        def warning_start():
            """Displays a warning when CogAgent starts processing."""
            return gr.Warning(
                "CogAgent is processing. Please do not interact with the keyboard or mouse.",
                duration=60,
            )

        # CogAgent完成处理时显示警告。
        def warning_end():
            """Displays a warning when CogAgent finishes processing."""
            return gr.Warning("CogAgent has finished. Please input a new task.")

        # Create a partial workflow function with fixed parameters
        # 创建固定参数的部分工作流函数, 其中api_key, base_url, model为固定参数
        workflow_partial = partial(workflow, api_key, base_url, model)

        # Button actions and callbacks
        submit_button.click(
            user_input, inputs=[task, chatbot], outputs=[task, chatbot], queue=False
        ).then(
            workflow_partial,
            inputs=[chatbot, max_length, top_p, temperature],
            outputs=[chatbot, img_path],
        ).then(
            warning_end, inputs=None, outputs=None
        )

        # 清理历史记录按钮
        clear_button.click(
            lambda: (None, None), inputs=None, outputs=[chatbot, img_path], queue=False
        )

        # 停止按钮
        stop_button.click(switch).then(raise_error, inputs=None, outputs=None)

    demo.queue()
    demo.launch(server_name=client_name, server_port=client_port)


stop_event = threading.Event()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the CogAgent demo with all parameters controlled by command-line arguments."
    )
    parser.add_argument("--api_key", default=API_token, help="OpenAI API Key.")
    parser.add_argument("--base_url", default=API_url, help="OpenAI API Base URL.")
    parser.add_argument("--model", default="cogagent-9b-20241220", help="Model name to use.")
    parser.add_argument(
        "--client_name",
        default="127.0.0.1",
        help="The IP or hostname for Gradio.",
    )
    parser.add_argument("--client_port", type=int, default=8088, help="The port number for Gradio.")

    args = parser.parse_args()

    if not os.path.exists("caches"):
        os.makedirs("caches")
    if not os.path.exists("screenshot"):
        os.makedirs("screenshot")

    gradio_web(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        client_name=args.client_name,
        client_port=args.client_port,
    )
