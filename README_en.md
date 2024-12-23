# CogAgent

阅读[中文](README.md)版本

## About the Model

### Model Introduction

`cogagent-9b-20241220` is a model specially used for Agent tasks that we trained based on [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b).
`cogagent-9b-20241220` is a relatively advanced intelligent agent model with strong cross-platform compatibility and can automate the operation of graphical interfaces on multiple computing devices.
Whether it is Windows, macOS or Android system, `cogagent-9b-20241220` can receive user instructions, automatically obtain device screenshots, and perform automated device operations after model inference.

### Model download links and hardware requirements

|     Model      |                                                                                               HuggingFace Model                                                                                                |     Inference Cost      |                                         Fine-tuning costs                                         |
|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------:|:------------------------------------------------------------------------------------:|
| cogagent-9b-20241220 | [🤗 Huggingface](https://huggingface.co/THUDM/cogagent-9b-20241220)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogagent-9b-20241220) <br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/cogagent-9b-20241220) | BF16: 29GB | LORA： 67GB (bs=1, 1 * GPU, 2048token context) <br> SFT: 55GB(Freeze vision, bs=1, 8 * GPU, zero3 data parallelism, 2048token context) |

+ The GPU in all the above data refers to A100 and H100 GPUs. The video memory/memory of other devices needs to be calculated by yourself.
+ It is not recommended to use `INT4` precision inference, as it will result in a significant performance loss.
+ Performance data and support on non-`NVIDIA` devices have not been tested.
+ The 1600 tokens taken by the image encoding are not counted in the context.

### Performance Data

Here are some of the rankings of the model:

## Input and Output

`cogagent-9b-20241220` is an Agent-like execution model rather than a dialogue model. It does not support continuous dialogue, but it supports continuous execution history.
This shows how users should format their input and pass it to the model, and get the response of the model rules.

### User Input Section

1. `task` field

   The task description entered by the user is similar to a text prompt. This input can guide the `cogagent-9b-20241220` model to complete the user's task instructions. Please keep it concise and clear.

2. `platform` field

   `cogagent-9b-20241220` supports executing operable Agent functions on multiple platforms. We support three operating systems with graphical interfaces:
    - Windows 10，11, Please use the `WIN` field.
    - Mac 14，15, Please use the `MAC` field.
    - For Android 13, 14, 15, and other Android UI distributions where the GUI and UI operations are almost the same, please use the `Mobile` field.

   If you are using another system, it may not work as well, but you can try using the `Mobile` field for mobile devices, the `WIN` field for Windows devices, and the `MAC` field for Mac devices.

3. `format` field

   What format do users want `cogagent-9b-20241220` to return data in? There are several options:
    - `Answer in Action-Operation-Sensitive format.`: The default return method used by the demo in this repository, the behavior of the return model, the corresponding operations, and the corresponding sensitivity.
    - `Answer in Status-Plan-Action-Operation format.`: Returns the model's configuration, behavior, and corresponding operations.
    - `Answer in Status-Action-Operation-Sensitive format.`: Returns the model's state, behavior, corresponding operations, and corresponding sensitivity.
    - `Answer in Status-Action-Operation format.`: Returns the state and behavior of the model.
    - `Answer in Action-Operation format.` Returns the behavior of the model, the corresponding operation.

4. `history` field

   The concatenation order and result should be as follows:
   ```
   query = f'{task}{history}{platform}{format}'
   ```

### Model return section

1. Sensitive operations: including `<<sensitive operation>> <<general operation>>`, and are only returned when `Sensitive` is required.
2. `Plan`, `Status`, `Action` fields: used to describe the behavior and operation of the model. They are only returned when the corresponding fields are required to be returned. For example, if there is an `Action`, the contents of the `Action` field are returned.
3. The general answer section will be before the formatted answer and will serve as a summary.
4. `Grounded Operation` field:
   It is used to describe the specific operation of the model, including the location, type, and specific operation content of the operation. Among them, `box` represents the coordinates of the execution area, and `element_type` represents the element type of the execution.
   `element_info` represents the element description of the execution. This information is wrapped by an `operation instruction` operation. Please refer to [here](Action_space_en.md) for more information about the action space.

### Examples

The user's task is to help mark all emails as read. The user is using a Mac system and hopes to be returned in the Action-Operation-Sensitive format.
The correct splicing prompt should be:

```
Task: Mark all my emails as read
(Platform: Mac)
(Answer in Action-Operation-Sensitive format.)
```

Next, here are the returned results under different format requirements:


<details>
<summary>Answer in Action-Operation-Sensitive format</summary>

```
Action: Click the "Mark All as Read" button in the toolbar at the top of the page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='clickable text', element_info='mark all as read')
<<General Operation>>
```

</details>

<details>
<summary>Answer in Status-Plan-Action-Operation format</summary>

```
Status: You are currently in the mailbox interface [[0, 2, 998, 905]], with mailbox categories on the left [[1, 216, 144, 570]] and the inbox in the middle [[144, 216, 998, 903]], and the "Mark All as Read" button [[223, 178, 311, 210]] has been clicked.
Action: Click the "Mark All as Read" button in the toolbar at the top of the page to mark all emails as read.
Plan: 1. click the "Mark All as Read" button; 2. task finished.
Action: Click the "Mark All as Read" button at the top center of your Inbox page to mark all messages as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='clickable text', element_info='mark all as read')
```

</details>

<details>
<summary>Answer in Status-Action-Operation-Sensitive format</summary>

```
Status: You are currently in the mailbox interface [[0, 2, 998, 905]], with mailbox categories on the left [[1, 216, 144, 570]] and the inbox in the middle [[144, 216, 998, 903]], and the "Mark All as Read" button [[223, 178, 311, 210]] has been clicked.
Action: Click the "Mark All as Read" button in the toolbar at the top of the page to mark all emails as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='clickable text', element_info='mark all as read')
<<General Operation>>
```

</details>

<details>
<summary>Answer in Status-Action-Operation format</summary>

```
Status: You are currently in the mailbox interface [[0, 2, 998, 905]], with mailbox categories on the left [[1, 216, 144, 570]] and the inbox in the middle [[144, 216, 998, 903]], and the "Mark All as Read" button [[223, 178, 311, 210]] has been clicked.
Action: Click the "Mark All as Read" button in the toolbar at the top of the page to mark all emails as read.
Action: At the top of the Inbox page, click the Mark All as Read button to mark all messages as read.
Grounded Operation: CLICK(box=[[219,186,311,207]], element_type='clickable text', element_info='mark all as read')
```

</details>

<details>
<summary>Answer in Action-Operation format</summary>

```
Action: In the message list on the left, right-click the first message to open the action menu.
Grounded Operation: RIGHT_CLICK(box=[[154,275,343,341]], element_info='[AXCell]')
```

</details>

### Precautions

1. This model is not a dialogue model and does not support continuous dialogue. Please send specific instructions and refer to the historical splicing method we provide for splicing.
2. This model must have images passed in, and pure text dialogue cannot achieve GUI Agent tasks.
3. The model output has strict format requirements. Please parse it strictly according to our requirements. The output format is STR format, and the output JSON format is not supported.

## Run the model

### Environment Configuration

Please make sure you have installed `Python 3.10.16` or above. And install the following dependencies:

```shell
pip install -r requirements.txt
```

To run a local transformers based model inference, you can run the model by running the following command:

```shell
python inference/cli_demo.py --model_dir THUDM/cogagent-9b-20241220 --platform "Mac" --max_length 4096 --top_k 1 --output_image_path ./results --format_key status_action_op_sensitive
```

This is a command line interactive code. You need to enter the corresponding image path. If the result returned by the model has a bbox, an image with a bbox will be output, indicating that the operation needs to be performed in this area. The saved image is in the path of `output_image_path`, and the image name is `{the image name you entered}_{dialogue turn}.png`. `format_key` indicates the format you want the model to return. The `platform` field determines which platform you serve (for example, `Mac`, then the screenshots you upload must be screenshots of the `Mac` system).

If you want to run the online web demo, this is a demo that requires continuous uploading of images for interaction, and the model will return the corresponding Bbox and execution category. This code has the same effect as the online experience of `HuggingFace Space`.

```shell
python inference/web_demo.py --host 0.0.0.0 --port 7860 --model_dir THUDM/cogagent-9b-20241220 --format_key status_action_op_sensitive --platform "Mac" --output_dir ./results
```

### Run the Agent APP Example

We have prepared a basic Demo APP for developers to demonstrate the GUI capabilities of the `cogagent-9b-20241220` model. The Demo shows how to deploy the model on a server with a GPU and run the `cogagent-9b-20241220` model on a local computer to perform automated GUI operations.

> We cannot guarantee the safety of AI's behavior, please use caution when using it.
>
> This example is for academic reference only, and we do not assume any legal liability caused by this example.

If you are interested in this APP, please check out the [Documentation](app/README.md)

### Fine-tuning the model

If you are interested in fine-tuning the `cogagent-9b-20241220` model, please check out [here](finetune/README.md).

## Previous Work

In November 2023, we released the first generation model of CogAgent. Now, you can find the relevant code and weight address in [CogVLM&CogAgent official repository](https://github.com/THUDM/CogVLM)

<div align="center">
    <img src=assets/cogagent_function_cn.jpg width=70% />
</div>

<table>
  <tr>
    <td>
      <h2> CogVLM </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2311.03079">CogVLM: Visual Expert for Pretrained Language Models</a></p>
      <p><b>CogVLM</b> is a powerful open source visual language model (VLM). CogVLM-17B has 10 billion visual parameters and 7 billion language parameters, supporting 490*490 resolution image understanding and multi-round dialogue.</p>
      <p><b>CogVLM-17B 17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks</b> including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC benchmarks.</p>
    </td>
    <td>
      <h2> CogAgent </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2312.08914">CogAgent: A Visual Language Model for GUI Agents </a></p>
      <p><b>CogAgent</b> is an open source visual language model based on CogVLM. CogAgent-18B has 11 billion visual parameters and 7 billion language parameters, <b>supports image understanding with a resolution of 1120*1120. In addition to the capabilities of CogVLM, it further possesses the capabilities of a GUI image agent.</b></p>
      <p> <b>CogAgent-18B achieves state-of-the-art general performance on 9 classic cross-modal benchmarks，</b>includes VQAv2, OK-VQ, TextVQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE test benchmarks. It significantly outperforms existing models on GUI operation datasets including AITW and Mind2Web.</p>
    </td>
  </tr>
</table>

## Protocol

The code in this github repository uses [Apache2.0 LICENSE](LICENSE).

Use of model weights is subject to the [Model License](MODEL_LICENSE).
