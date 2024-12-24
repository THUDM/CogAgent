# CogAgent

## About the Model

### Model Overview

`cogagent-9b-20241220` is a model we have trained based on [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b), designed
specifically for agent tasks.  
`cogagent-9b-20241220` is an advanced agent model that features strong cross-platform compatibility, enabling it to
automate graphical interface operations on a variety of computing devices. Whether on Windows, macOS, or Android,
`cogagent-9b-20241220` can receive user instructions, automatically capture device screenshots, perform model inference,
and execute automated device operations.

### Capability Demonstrations

<div style="display: flex; flex-direction: column; width: 100%; align-items: center; margin-top: 20px;">
    <div style="text-align: center; margin-bottom: 20px; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/9309de98-3051-4e78-b64f-da501d624d66" width="100%" height="auto" controls autoplay loop></video>
        <p>CogAgent wishes you a Merry Christmas! Let the large model automatically send Christmas greetings to your friends.</p>
    </div>
    <div style="text-align: center; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/53f6c39b-3785-433a-aaf8-475124dad35d" width="100%" height="auto" controls autoplay loop></video>
        <p>Want to open an issue? Let CogAgent help you send an email.</p>
    </div>
</div>

### Model Download Links

|        Model         |                                                                                                          Download Link                                                                                                           |                    Download Link (Ascend Optimized)                    | Try it Online                                                                      |
|:--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------:|------------------------------------------------------------------------------------|
| cogagent-9b-20241220 | [🤗 HuggingFace](https://huggingface.co/THUDM/cogagent-9b-20241220)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogagent-9b-20241220) <br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/cogagent-9b-20241220) | [🧩 Modelers](https://modelers.cn/models/zhipuai/cogagent-9b-20241220) | [🚀 HuggingFace Space](https://huggingface.co/spaces/THUDM-HF-SPACE/CogAgent-Demo) |

### Inference and Fine-tuning Costs

+ At `BF16` precision, the model requires **at least** `29GB` of GPU memory for inference. Using `INT4` precision is *
  *not** recommended due to significant performance loss.
+ All GPU references above refer to A100 or H100 GPUs. For other devices, you need to calculate the required GPU/CPU
  memory accordingly.
+ During SFT (Supervised Fine-Tuning), this codebase freezes the `Vision Encoder`, uses a batch size of 1, and trains on
  `8 * A100` GPUs. The total input tokens (including images, which account for `1600` tokens) add up to 2048 tokens.
  This codebase cannot conduct SFT fine-tuning without freezing the `Vision Encoder`.  
  For LoRA fine-tuning, `Vision Encoder` is **not** frozen; the batch size is 1, using `1 * A100` GPU. The total input
  tokens (including images, `1600` tokens) also amount to 2048 tokens. In the above setup, SFT fine-tuning requires at
  least `60GB` of GPU memory per GPU (with 8 GPUs), while LoRA fine-tuning requires at least `70GB` of GPU memory on a
  single GPU (cannot be split).
+ `Ascend devices` have not been tested for SFT fine-tuning. We have only tested them on the `Atlas800` training server
  cluster. You need to modify the inference code accordingly based on the loading mechanism described in the
  `Ascend device` download link.
+ Currently, we do **not** support inference with the `vLLM` framework. We will submit a PR as soon as possible to
  enable it.
+ The online demo link does **not** support controlling computers; it only allows you to view the model's inference
  results. We recommend deploying the model locally.

## Inputs and Outputs

`cogagent-9b-20241220` is an agent-type execution model rather than a conversational model. It does not support
continuous dialogue, but it **does** support a continuous execution history. Below is how users should format their
inputs and feed them to the model, and how to interpret the model’s responses.

### User Input

1. **`task` field**  
   The user’s task description, in text format similar to a prompt. This input instructs the `cogagent-9b-20241220`
   model on how to carry out the user’s request. Keep it concise and clear.

2. **`platform` field**  
   `cogagent-9b-20241220` supports agent operations on multiple platforms with graphical interfaces. We currently
   support three systems:
    - Windows 10, 11: Use the `WIN` field.
    - macOS 14, 15: Use the `Mac` field.
    - Android 13, 14, 15 (and other Android UI variants with similar GUI operations): Use the `Mobile` field.

   If your system is not among these, the effectiveness may be suboptimal. You can try using `Mobile` for mobile
   devices, `WIN` for Windows, or `Mac` for Mac.

3. **`format` field**  
   The format in which the user wants `cogagent-9b-20241220` to return data. We provide several options:
    - `Answer in Action-Operation-Sensitive format.`: The default demo return type in this repo. Returns the model’s
      actions, corresponding operations, and the sensitivity level.
    - `Answer in Status-Plan-Action-Operation format.`: Returns the model’s status, plan, and corresponding operations.
    - `Answer in Status-Action-Operation-Sensitive format.`: Returns the model’s status, actions, corresponding
      operations, and sensitivity.
    - `Answer in Status-Action-Operation format.`: Returns the model’s status and actions.
    - `Answer in Action-Operation format.`: Returns the model’s actions and corresponding operations.

4. **`history` field**  
   This should be concatenated in the following order:
   ```
   query = f'{task}{history}{platform}{format}'
   ```

### Model Output

1. **Sensitive operations**: Includes `<<敏感操作>>` (“sensitive operation”) and `<<一般操作>>` (“general operation”).
   These are only returned if you request the `Sensitive` format.
2. **`Plan`, `Status`, `Action` fields**: Used to describe the model’s behavior and operations. Only returned if you
   request the corresponding fields. For example, if the format includes `Action`, then the model returns the `Action`
   field.
3. **General answer section**: A summary that appears prior to the formatted answer.
4. **`Grounded Operation` field**:  
   Describes the model’s specific operations, including the location of the operation, the operation type, and the
   action details. The `box` attribute indicates the coordinate region for execution, `element_type` indicates the
   element type, and `element_info` describes the element. These details are wrapped within a “操作指令” (operation
   command). For the definition of the action space, please refer to [here](Action_space.md).

### Example

Suppose the user wants to mark all emails as read. The user is on a Mac, and the user wants the model to return in
`Action-Operation-Sensitive` format. The properly concatenated prompt should be:

```
Task: 帮我将所有的邮件标注为已读
(Platform: Mac)
(Answer in Action-Operation-Sensitive format.)
```

Below are sample outputs for different format requirements:

<details>
<summary>Answer in Action-Operation-Sensitive format</summary>

```
Action: 点击页面顶部工具栏中的“全部标为已读”按钮，将所有邮件标记为已读。
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='可点击文本', element_info='全部标为已读')
<<一般操作>>
```

</details>

<details>
<summary>Answer in Status-Plan-Action-Operation format</summary>

```
Status: 当前处于邮箱界面[[0, 2, 998, 905]]，左侧是邮箱分类[[1, 216, 144, 570]]，中间是收件箱[[144, 216, 998, 903]]，已经点击“全部标为已读”按钮[[223, 178, 311, 210]]。
Plan: 未来需要: 1. 点击“全部标记为已读”按钮；2. 任务结束。
Action: 点击收件箱页面顶部中间的“全部标记为已读”按钮，将所有邮件标记为已读。
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='可点击文本', element_info='全部标为已读')
```

</details>

<details>
<summary>Answer in Status-Action-Operation-Sensitive format</summary>

```
Status: 当前处于邮箱界面[[0, 2, 998, 905]]，左侧是邮箱分类[[1, 216, 144, 570]]，中间是收件箱[[144, 216, 998, 903]]，已经点击“全部标为已读”按钮[[223, 178, 311, 210]]。
Action: 点击页面顶部工具栏中的“全部标为已读”按钮，将所有邮件标记为已读。
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='可点击文本', element_info='全部标为已读')
<<一般操作>>
```

</details>

<details>
<summary>Answer in Status-Action-Operation format</summary>

```
Status: 当前处于邮箱界面[[0, 2, 998, 905]]，左侧是邮箱分类[[1, 216, 144, 570]]，中间是收件箱[[144, 216, 998, 903]]，已经点击“全部标为已读”按钮[[223, 178, 311, 210]]。
Action: 在收件箱页面顶部，点击“全部标记为已读”按钮，将所有邮件标记为已读。
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='可点击文本', element_info='全部标为已读')
```

</details>

<details>
<summary>Answer in Action-Operation format</summary>

```
Action: 在左侧邮件列表中，右键单击第一封邮件，以打开操作菜单。
Grounded Operation: RIGHT_CLICK(box=[[154,275,343,341]], element_info='[AXCell]')
```

</details>

### Notes

1. This model is **not** a conversational model and does **not** support continuous dialogue. Please send specific
   commands and reference our recommended method for concatenating the history.
2. The model **requires** images as input; pure text conversation cannot achieve GUI Agent tasks.
3. The model’s output adheres to a strict format. Please parse it strictly according to our requirements. The output is
   in **string** format; JSON output is **not** supported.

## Running the Model

### Environment Setup

Make sure you have installed **Python 3.10.16** or above, and then install the following dependencies:

```shell
pip install -r requirements.txt
```

To run local inference based on `transformers`, you can run the command below:

```shell
python inference/cli_demo.py --model_dir THUDM/cogagent-9b-20241220 --platform "Mac" --max_length 4096 --top_k 1 --output_image_path ./results --format_key status_action_op_sensitive
```

This is a command-line interactive code. You will need to provide the path to your images. If the model returns results
containing bounding boxes, it will output an image with those bounding boxes, indicating the region where the operation
should be executed. The image is saved to `output_image_path`, with the file name `{your_input_image_name}_{round}.png`.
The `format_key` indicates in which format you want the model to respond. The `platform` field specifies which platform
you are using (e.g., `Mac`). Therefore, all uploaded screenshots must be from macOS if `platform` is set to `Mac`.

If you want to run an online web demo, which supports continuous image uploads for interactive inference, you can run:

```shell
python inference/web_demo.py --host 0.0.0.0 --port 7860 --model_dir THUDM/cogagent-9b-20241220 --format_key status_action_op_sensitive --platform "Mac" --output_dir ./results
```

This code provides the same experience as the `HuggingFace Space` online demo. The model will return the corresponding
bounding boxes and execution categories.

### Running an Agent APP Example

We have prepared a basic demo app for developers to illustrate the GUI capabilities of `cogagent-9b-20241220`. The demo
shows how to deploy the model on a GPU-equipped server and run the `cogagent-9b-20241220` model locally to perform
automated GUI operations.

> We cannot guarantee the safety of AI behavior; please exercise caution when using it.  
> This example is only for academic reference. We assume no legal responsibility for any issues resulting from this
> example.

If you are interested in this APP, feel free to check out the [documentation](app/README.md).

### Fine-tuning the Model

If you are interested in fine-tuning the `cogagent-9b-20241220` model, please refer to [here](finetune/README.md).

## Previous Work

In November 2023, we released the first generation of CogAgent. You can find related code and model weights in
the [CogVLM & CogAgent Official Repository](https://github.com/THUDM/CogVLM).

<div align="center">
    <img src=assets/cogagent_function_cn.jpg width=70% />
</div>

<table>
  <tr>
    <td>
      <h2> CogVLM </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2311.03079">CogVLM: Visual Expert for Pretrained Language Models</a></p>
      <p><b>CogVLM</b> is a powerful open-source Vision-Language Model (VLM). CogVLM-17B has 10B visual parameters and 7B language parameters, supporting image understanding at a resolution of 490x490, as well as multi-round dialogue.</p>
      <p><b>CogVLM-17B</b> achieves state-of-the-art performance on 10 classic multimodal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA, and TDIUC.</p>
    </td>
    <td>
      <h2> CogAgent </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2312.08914">CogAgent: A Visual Language Model for GUI Agents</a></p>
      <p><b>CogAgent</b> is an open-source vision-language model improved upon CogVLM. CogAgent-18B has 11B visual parameters and 7B language parameters. <b>It supports image understanding at a resolution of 1120x1120. Building on CogVLM’s capabilities, CogAgent further incorporates a GUI image agent ability.</b></p>
      <p><b>CogAgent-18B</b> delivers state-of-the-art general performance on 9 classic vision-language benchmarks, including VQAv2, OK-VQ, TextVQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. It also significantly outperforms existing models on GUI operation datasets such as AITW and Mind2Web.</p>
    </td>
  </tr>
</table>

## License

- The [Apache2.0 LICENSE](LICENSE) applies to the use of the code in this GitHub repository.
- For the model weights, please follow the [Model License](MODEL_LICENSE).  
