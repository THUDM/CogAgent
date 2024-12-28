import argparse
import os
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List


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


def main():
    """
    A continuous interactive demo using the CogAgent1.5 model with selectable format prompts.
    The output_image_path is interpreted as a directory. For each round of interaction,
    the annotated image will be saved in the directory with the filename:
    {original_image_name_without_extension}_{round_number}.png

    Example:
    python cli_demo.py --model_dir THUDM/cogagent-9b-20241220 --platform "Mac" --max_length 4096 --top_k 1 \
                     --output_image_path ./results --format_key status_action_op_sensitive
    """

    parser = argparse.ArgumentParser(
        description="Continuous interactive demo with CogAgent model and selectable format."
    )
    parser.add_argument(
        "--model_dir", required=True, help="Path or identifier of the model."
    )
    parser.add_argument(
        "--platform",
        default="Mac",
        help="Platform information string (e.g., 'Mac', 'WIN').",
    )
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum generation length."
    )
    parser.add_argument(
        "--top_k", type=int, default=1, help="Top-k sampling parameter."
    )
    parser.add_argument(
        "--output_image_path",
        default="results",
        help="Directory to save the annotated images.",
    )
    parser.add_argument(
        "--format_key",
        default="action_op_sensitive",
        help="Key to select the prompt format.",
    )
    args = parser.parse_args()

    # Dictionary mapping format keys to format strings
    format_dict = {
        "action_op_sensitive": "(Answer in Action-Operation-Sensitive format.)",
        "status_plan_action_op": "(Answer in Status-Plan-Action-Operation format.)",
        "status_action_op_sensitive": "(Answer in Status-Action-Operation-Sensitive format.)",
        "status_action_op": "(Answer in Status-Action-Operation format.)",
        "action_op": "(Answer in Action-Operation format.)",
    }

    # Ensure the provided format_key is valid
    if args.format_key not in format_dict:
        raise ValueError(
            f"Invalid format_key. Available keys are: {list(format_dict.keys())}"
        )

    # Ensure the output directory exists
    os.makedirs(args.output_image_path, exist_ok=True)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True), # For INT8 quantization
        # quantization_config=BitsAndBytesConfig(load_in_4bit=True), # For INT4 quantization
    ).eval()
    # Initialize platform and selected format strings
    platform_str = f"(Platform: {args.platform})\n"
    format_str = format_dict[args.format_key]

    # Initialize history lists
    history_step = []
    history_action = []

    round_num = 1
    while True:
        task = input("Please enter the task description ('exit' to quit): ")
        if task.lower() == "exit":
            break

        img_path = input("Please enter the image path: ")
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            print("Invalid image path. Please try again.")
            continue

        # Verify history lengths match
        if len(history_step) != len(history_action):
            raise ValueError("Mismatch in lengths of history_step and history_action.")

        # Format history steps for output
        history_str = "\nHistory steps: "
        for index, (step, action) in enumerate(zip(history_step, history_action)):
            history_str += f"\n{index}. {step}\t{action}"

        # Compose the query with task, platform, and selected format instructions
        query = f"Task: {task}{history_str}\n{platform_str}{format_str}"

        print(f"Round {round_num} query:\n{query}")

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        # Generation parameters
        gen_kwargs = {
            "max_length": args.max_length,
            "do_sample": True,
            "top_k": args.top_k,
        }

        # Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model response:\n{response}")

        # Extract grounded operation and action
        grounded_pattern = r"Grounded Operation:\s*(.*)"
        action_pattern = r"Action:\s*(.*)"
        matches_history = re.search(grounded_pattern, response)
        matches_actions = re.search(action_pattern, response)

        if matches_history:
            grounded_operation = matches_history.group(1)
            history_step.append(grounded_operation)
        if matches_actions:
            action_operation = matches_actions.group(1)
            history_action.append(action_operation)

        # Extract bounding boxes from the response
        box_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
        matches = re.findall(box_pattern, response)
        if matches:
            boxes = [[int(x) / 1000 for x in match] for match in matches]

            # Extract base name of the user's input image (without extension)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Construct the output file name with round number
            output_file_name = f"{base_name}_{round_num}.png"
            output_path = os.path.join(args.output_image_path, output_file_name)

            draw_boxes_on_image(image, boxes, output_path)
            print(f"Annotated image saved at: {output_path}")
        else:
            print("No bounding boxes found in the response.")

        round_num += 1


if __name__ == "__main__":
    main()
