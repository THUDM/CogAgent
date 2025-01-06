# CogAgent: An open-sourced VLM-based GUI Agent

[中文文档](README_zh.md)

- 🔥 🆕 **December 2024:** We open-sourced **the latest version of the CogAgent-9B-20241220 model**. Compared to the
  previous version of CogAgent, `CogAgent-9B-20241220` features significant improvements in GUI perception, reasoning
  accuracy, action space completeness, task universality, and generalization. It supports bilingual (Chinese and
  English) interaction through both screen captures and natural language.

- 🏆 **June 2024:** CogAgent was accepted by **CVPR 2024** and recognized as a conference Highlight (top 3%).

- **December 2023:** We **open-sourced the first GUI Agent**: **CogAgent** (with the former repository
  available [here](https://github.com/THUDM/CogVLM)) and **published the corresponding paper:
  📖 [CogAgent Paper](https://arxiv.org/abs/2312.08914)**.

## Model Introduction

|        Model         |                                                                                                                                                 Model Download Links                                                                                                                                                 | Technical Documentation                                                                                                                                                                                                               | Online Demo                                                                                                                                                                                                                                     |                                                          
|:--------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|   
| cogagent-9b-20241220 | [🤗 HuggingFace](https://huggingface.co/THUDM/cogagent-9b-20241220)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogagent-9b-20241220) <br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/cogagent-9b-20241220) <br>[🧩 Modelers (Ascend)](https://modelers.cn/models/zhipuai/cogagent-9b-20241220) | [📄 Official Technical Blog](https://cogagent.aminer.cn/blog#/articles/cogagent-9b-20241220-technical-report)<br/>[📘 Practical Guide (Chinese)](https://zhipu-ai.feishu.cn/wiki/MhPYwtpBhinuoikNIYYcyu8dnKv?fromScene=spaceOverview) | [🤗 HuggingFace Space](https://huggingface.co/spaces/THUDM-HF-SPACE/CogAgent-Demo)<br/>[🤖 ModelScope Space](https://modelscope.cn/studios/ZhipuAI/CogAgent-Demo)<br/>[🧩 Modelers Space (Ascend)](https://modelers.cn/spaces/zhipuai/CogAgent) |

### Model Overview

`CogAgent-9B-20241220` model is based on [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b), a bilingual open-source
VLM base model. Through data collection and optimization, multi-stage training, and strategy improvements,
`CogAgent-9B-20241220` achieves significant advancements in GUI perception, inference prediction accuracy, action space
completeness, and generalizability across tasks. The model supports bilingual (Chinese and English) interaction with
both screenshots and language input. This version of the CogAgent model has already been applied in
ZhipuAI's [GLM-PC product](https://cogagent.aminer.cn/home). We hope the release of this model can assist researchers
and developers in advancing the research and applications of GUI agents based on vision-language models.

### Capability Demonstrations

The CogAgent-9b-20241220 model has achieved state-of-the-art results across multiple platforms and categories in GUI
Agent tasks and GUI Grounding Benchmarks. In
the [CogAgent-9b-20241220 Technical Blog](https://cogagent.aminer.cn/blog#/articles/cogagent-9b-20241220-technical-report),
we compared it against API-based commercial models (GPT-4o-20240806, Claude-3.5-Sonnet), commercial API + GUI Grounding
models (GPT-4o + UGround, GPT-4o + OS-ATLAS), and open-source GUI Agent models (Qwen2-VL, ShowUI, SeeClick). The results
demonstrate that **CogAgent leads in GUI localization (Screenspot), single-step operations (OmniAct), the Chinese
step-wise in-house benchmark (CogAgentBench-basic-cn), and multi-step operations (OSWorld)**, with only a slight
disadvantage in OSWorld compared to Claude-3.5-Sonnet, which specializes in Computer Use, and GPT-4o combined with
external GUI Grounding models.

<div style="display: flex; flex-direction: column; width: 100%; align-items: center; margin-top: 20px;">
    <div style="text-align: center; margin-bottom: 20px; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/4d39fe6a-d460-427c-a930-b7cbe0d082f5" width="100%" height="auto" controls autoplay loop></video>
        <p>CogAgent wishes you a Merry Christmas! Let the large model automatically send Christmas greetings to your friends.</p>
    </div>
    <div style="text-align: center; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/87f00f97-1c4f-4152-b7c0-d145742cb910" width="100%" height="auto" controls autoplay loop></video>
        <p>Want to open an issue? Let CogAgent help you send an email.</p>
    </div>
</div>


**Table of Contents**

- [CogAgent](#cogagent)
    - [Model Introduction](#model-introduction)
        - [Model Overview](#model-overview)
        - [Capability Demonstrations](#capability-demonstrations)
        - [Inference and Fine-tuning Costs](#inference-and-fine-tuning-costs)
    - [Model Inputs and Outputs](#model-inputs-and-outputs)
        - [User Input](#user-input)
        - [Model Output](#model-output)
        - [An Example](#an-example)
        - [Notes](#notes)
    - [Running the Model](#running-the-model)
        - [Environment Setup](#environment-setup)
        - [Running an Agent APP Example](#running-an-agent-app-example)
        - [Fine-tuning the Model](#fine-tuning-the-model)
    - [Previous Work](#previous-work)
    - [License](#license)
    - [Citation](#citation)
    - [Research and Development Team \& Acknowledgements](#research-and-development-team---acknowledgements)

### Inference and Fine-tuning Costs

+ The model requires at least 29GB of VRAM for inference at `BF16` precision. Using `INT4` precision for inference is
  not recommended due to significant performance loss. The VRAM usage for `INT4` inference is about 8GB, while for
  `INT8` inference it is about 15GB. In the `inference/cli_demo.py` file, we have commented out these two lines. You can
  uncomment them and use `INT4` or `INT8` inference. This solution is only supported on NVIDIA devices.
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

## Model Inputs and Outputs

`cogagent-9b-20241220` is an agent-type execution model rather than a conversational model. It does not support
continuous dialogue, but it **does** support a continuous execution history. (In other words, each time a new
conversation session needs to be started, and the past history should be provided to the model.) The workflow of
CogAgent is illustrated as following:

<div align="center">
    <img src=assets/cogagent_workflow_en.png width=90% />
</div>

**To achieve optimal GUI Agent performance, we have adopted a strict input-output format.**
Below is how users should format their inputs and feed them to the model, and how to interpret the model’s responses.

### User Input

You can refer
to [app/client.py#L115](https://github.com/THUDM/CogAgent/blob/e3ca6f4dc94118d3dfb749f195cbb800ee4543ce/app/client.py#L115)
for constructing user input prompts. A minimal example of user input concatenation code is shown below:

``` python

current_platform = identify_os() # "Mac" or "WIN" or "Mobile". Pay attention to case sensitivity.
platform_str = f"(Platform: {current_platform})\n"
format_str = "(Answer in Action-Operation-Sensitive format.)\n" # You can use other format to replace "Action-Operation-Sensitive"

history_str = "\nHistory steps: "
for index, (grounded_op_func, action) in enumerate(zip(history_grounded_op_funcs, history_actions)):
   history_str += f"\n{index}. {grounded_op_func}\t{action}" # start from 0. 

query = f"Task: {task}{history_str}\n{platform_str}{format_str}" # Be careful about the \n

```

The concatenated Python string:

``` python
"Task: Search for doors, click doors on sale and filter by brands \"Mastercraft\".\nHistory steps: \n0. CLICK(box=[[352,102,786,139]], element_info='Search')\tLeft click on the search box located in the middle top of the screen next to the Menards logo.\n1. TYPE(box=[[352,102,786,139]], text='doors', element_info='Search')\tIn the search input box at the top, type 'doors'.\n2. CLICK(box=[[787,102,809,139]], element_info='SEARCH')\tLeft click on the magnifying glass icon next to the search bar to perform the search.\n3. SCROLL_DOWN(box=[[0,209,998,952]], step_count=5, element_info='[None]')\tScroll down the page to see the available doors.\n4. CLICK(box=[[280,708,710,809]], element_info='Doors on Sale')\tClick the \"Doors On Sale\" button in the middle of the page to view the doors that are currently on sale.\n(Platform: WIN)\n(Answer in Action-Operation format.)\n"
```

Printed prompt:
>
> Task: Search for doors, click doors on sale and filter by brands "Mastercraft".
>
> History steps:
>
> 0. CLICK(box=[[352,102,786,139]], element_info='Search')  Left click on the search box located in the middle top of
     the screen next to the Menards logo.
> 1. TYPE(box=[[352,102,786,139]], text='doors', element_info='Search') In the search input box at the top, type '
     doors'.
> 2. CLICK(box=[[787,102,809,139]], element_info='SEARCH')  Left click on the magnifying glass icon next to the search
     bar to perform the search.
> 3. SCROLL_DOWN(box=[[0,209,998,952]], step_count=5, element_info='[None]')  Scroll down the page to see the available
     doors.
> 4. CLICK(box=[[280,708,710,809]], element_info='Doors on Sale') Click the "Doors On Sale" button in the middle of the
     page to view the doors that are currently on sale.
>    
>
> (Platform: WIN)
>
> (Answer in Action-Operation format.)
>

If you want to understand the meaning and representation of each field in detail, please continue reading or refer to
the [Practical Documentation (in Chinese), "Prompt Concatenation" section](https://zhipu-ai.feishu.cn/wiki/D9FTwQ78fitS3CkZHUjcKEWTned).

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
   
5. **`Continue` field**  
   CogAgent allows users to let the model `continue answering`. This requires users to append the `[Continue]\n` field after `{task}`. In such cases, the concatenation sequence and result should be as follows:
   ```
   query = f'{task}[Continue]\n{history}{platform}{format}'
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

### An Example

Suppose the user wants to mark all emails as read. The user is on a Mac, and the user wants the model to return in
`Action-Operation-Sensitive` format. The properly **concatenated prompt** should be:

```
Task: Please mark all my emails as read
History steps:
(Platform: Mac)
(Answer in Action-Operation-Sensitive format.)
```

Note: even if there are no historical actions, "History steps:" still needs to be appended in the prompt. Below are *
*sample outputs** for different format requirements:

<details>
<summary>Answer in Action-Operation-Sensitive format</summary>

```
Action: Click the 'Mark all as read' button in the top toolbar of the page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='Clickable text', element_info='Mark all emails as read')
<<一般操作>>
```

</details>

<details>
<summary>Answer in Status-Plan-Action-Operation format</summary>

```
Status: Currently in the email interface [[0, 2, 998, 905]], with the email categories on the left [[1, 216, 144, 570]], and the inbox in the center [[144, 216, 998, 903]]. The "Mark all as read" button has been clicked [[223, 178, 311, 210]].
Plan: Future tasks: 1. Click the 'Mark all as read' button; 2. Task complete.
Action: Click the "Mark all as read" button at the top center of the inbox page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='Clickable text', element_info='Mark all emails as read')
```

</details>

<details>
<summary>Answer in Status-Action-Operation-Sensitive format</summary>

```
Status: Currently in the email interface [[0, 2, 998, 905]], with the email categories on the left [[1, 216, 144, 570]], and the inbox in the center [[144, 216, 998, 903]]. The "Mark all as read" button has been clicked [[223, 178, 311, 210]].
Action: Click the "Mark all as read" button at the top center of the inbox page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='Clickable text', element_info='Mark all emails as read')
<<一般操作>>
```

</details>

<details>
<summary>Answer in Status-Action-Operation format</summary>

```
Status: Currently in the email interface [[0, 2, 998, 905]], with the email categories on the left [[1, 216, 144, 570]], and the inbox in the center [[144, 216, 998, 903]]. The "Mark all as read" button has been clicked [[223, 178, 311, 210]].
Action: Click the "Mark all as read" button at the top center of the inbox page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='Clickable text', element_info='Mark all emails as read')
```

</details>

<details>
<summary>Answer in Action-Operation format</summary>

```
Action: Right-click the first email in the left email list to open the action menu.
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
    <img src=assets/cogagent_function.jpg width=70% />
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

## Citation

If you find our work helpful, please consider citing the following papers

```
@misc{hong2023cogagent,
      title={CogAgent: A Visual Language Model for GUI Agents}, 
      author={Wenyi Hong and Weihan Wang and Qingsong Lv and Jiazheng Xu and Wenmeng Yu and Junhui Ji and Yan Wang and Zihan Wang and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2312.08914},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

## Research and Development Team & Acknowledgements

**R&D Institutions**: Tsinghua University, Zhipu AI

**Team members**: Wenyi Hong, Junhui Ji, Lihang Pan, Yuanchang Yue, Changyu Pang, Siyan Xue, Guo Wang, Weihan Wang,
Jiazheng Xu, Shen Yang, Xiaotao Gu, Yuxiao Dong, Jie Tang

**Acknowledgement**: We would like to thank the Zhipu AI data team for their strong support, including Xiaohan Zhang,
Zhao Xue, Lu Chen, Jingjie Du, Siyu Wang, Ying Zhang, and all annotators. They worked hard to collect and annotate the
training and testing data of the CogAgent model. We also thank Yuxuan Zhang, Xiaowei Hu, and Hao Chen from the Zhipu AI
open source team for their engineering efforts in open sourcing the model.
