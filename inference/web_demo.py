import argparse
import os
import re
import torch
from threading import Thread, Event
from PIL import Image, ImageDraw
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from typing import List
import spaces

stop_event = Event()

def draw_boxes_on_image(image: Image.Image, boxes: List[List[float]], save_path: str):
    """
    Draws red bounding boxes on the given image and saves it.

    Parameters:
    - image (PIL.Image.Image): The image on which to draw the bounding boxes.
    - boxes (List[List[float]]): A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].
      Coordinates are expected to be normalized (0 to 1).
    - save_path (str): The path to save the updated image.

    Description:
    Each box coordinate is a fraction of the image dimension. This function converts them to actual pixel
    coordinates and draws a red rectangle to mark the area. The annotated image is then saved to the specified path.
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x_min = int(box[0] * image.width)
        y_min = int(box[1] * image.height)
        x_max = int(box[2] * image.width)
        y_max = int(box[3] * image.height)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    image.save(save_path)


def preprocess_messages(history, img_path, platform_str, format_str):
    history_step = []
    for task, model_msg in history:
        grounded_pattern = r"Grounded Operation:\s*(.*)"
        matches_history = re.search(grounded_pattern, model_msg)
        if matches_history:
            grounded_operation = matches_history.group(1)
            history_step.append(grounded_operation)

    history_str = "\nHistory steps: "
    if history_step:
        for i, step in enumerate(history_step):
            history_str += f"\n{i}. {step}"

    if history:
        task = history[-1][0]
    else:
        task = "No task provided"

    query = f"Task: {task}{history_str}\n{platform_str}{format_str}"
    image = Image.open(img_path).convert("RGB")
    return query, image


@spaces.GPU()
def predict(history, max_length, img_path, platform_str, format_str, output_dir):
    # Reset the stop_event at the start of prediction
    stop_event.clear()

    # Remember history length before this round (for rollback if stopped)
    prev_len = len(history)

    query, image = preprocess_messages(history, img_path, platform_str, format_str)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": inputs["position_ids"],
        "images": inputs["images"],
        "streamer": streamer,
        "max_length": max_length,
        "do_sample": True,
        "top_k": 1,
    }
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    with torch.no_grad():
        for new_token in streamer:
            # Check if stop event is set
            if stop_event.is_set():
                # Stop generation immediately
                # Rollback the last round user input
                while len(history) > prev_len:
                    history.pop()
                yield history, None
                return

            if new_token:
                history[-1][1] += new_token
            yield history, None


    response = history[-1][1]
    box_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
    matches = re.findall(box_pattern, response)
    if matches:
        boxes = [[int(x) / 1000 for x in match] for match in matches]
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        round_num = sum(1 for (u, m) in history if u and m)
        output_path = os.path.join(output_dir, f"{base_name}_{round_num}.png")
        image = Image.open(img_path).convert("RGB")
        draw_boxes_on_image(image, boxes, output_path)
        yield history, output_path
    else:
        yield history, None


def user(task, history):
    return "", history + [[task, ""]]


def undo_last_round(history, output_img):
    if history:
        history.pop()
    return history, None


def clear_all_history():
    return None, None


def stop_now():
    # Set the stop event to interrupt generation
    stop_event.set()
    # Returning no changes here, the changes to history and output_img are handled in predict
    return gr.update(), gr.update()


def main():
    parser = argparse.ArgumentParser(description="CogAgent Gradio Demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP for the server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the server.")
    parser.add_argument("--model_dir", required=True, help="Path or identifier of the model.")
    parser.add_argument("--format_key", default="action_op_sensitive", help="Key to select the prompt format.")
    parser.add_argument("--platform", default="Mac", help="Platform information string.")
    parser.add_argument("--output_dir", default="annotated_images", help="Directory to save annotated images.")
    args = parser.parse_args()

    format_dict = {
        "action_op_sensitive": "(Answer in Action-Operation-Sensitive format.)",
        "status_plan_action_op": "(Answer in Status-Plan-Action-Operation format.)",
        "status_action_op_sensitive": "(Answer in Status-Action-Operation-Sensitive format.)",
        "status_action_op": "(Answer in Status-Action-Operation format.)",
        "action_op": "(Answer in Action-Operation format.)"
    }

    if args.format_key not in format_dict:
        raise ValueError(f"Invalid format_key. Available keys: {list(format_dict.keys())}")

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    ).eval()

    platform_str = f"(Platform: {args.platform})\n"
    format_str = format_dict[args.format_key]

    with gr.Blocks(analytics_enabled=False) as demo:
        gr.HTML("<h1 align='center'>CogAgent Demo</h1>")
        gr.HTML(
            "<p align='center' style='color:red;'>This Demo is for learning and communication purposes only. Users must assume responsibility for the risks associated with AI-generated planning and operations.</p>")

        with gr.Row():
            img_path = gr.Image(label="Upload a Screenshot", type="filepath", height=400)
            output_img = gr.Image(type="filepath", label="Annotated Image", height=400, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=300)
                task = gr.Textbox(show_label=True, placeholder="Input...", label="Task")
                submitBtn = gr.Button("Submit")
            with gr.Column(scale=1):
                max_length = gr.Slider(0, 8192, value=1024, step=1.0, label="Maximum length", interactive=True)
                undo_last_round_btn = gr.Button("Back to Last Round")
                clear_history_btn = gr.Button("Clear All History")

                # Interrupt procedure
                stop_now_btn = gr.Button("Stop Now", variant="stop")

        submitBtn.click(
            user, [task, chatbot], [task, chatbot], queue=False
        ).then(
            predict,
            [chatbot, max_length, img_path, gr.State(platform_str), gr.State(format_str),
             gr.State(args.output_dir)],
            [chatbot, output_img],
            queue=True
        )

        undo_last_round_btn.click(undo_last_round, [chatbot, output_img], [chatbot, output_img], queue=False)
        clear_history_btn.click(clear_all_history, None, [chatbot, output_img], queue=False)
        stop_now_btn.click(stop_now, None, [chatbot, output_img], queue=False)

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
