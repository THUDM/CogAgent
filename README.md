# CogAgent

Read this in [English](README_en.md)

## 关于模型

### 模型资源

|        Model         |                                                                                                                                                       模型下载地址                                                                                                                                                        | 技术文档                                                                                                                                                                                              | 在线体验                                                                               |                                                          
|:--------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|   
| cogagent-9b-20241220 | [🤗 HuggingFace](https://huggingface.co/THUDM/cogagent-9b-20241220)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/cogagent-9b-20241220) <br> [🟣 WiseModel](https://wisemodel.cn/models/ZhipuAI/cogagent-9b-20241220) <br>[🧩 Modelers](https://modelers.cn/models/zhipuai/cogagent-9b-20241220)(昇腾设备优化) | [📄 技术报告](https://cogagent.aminer.cn/blog#/articles/cogagent-9b-20241220-technical-report)<br/>[📘 智谱AI技术文档](https://zhipu-ai.feishu.cn/wiki/MhPYwtpBhinuoikNIYYcyu8dnKv?fromScene=spaceOverview) | [🚀 HuggingFace Space](https://huggingface.co/spaces/THUDM-HF-SPACE/CogAgent-Demo) | 

### 模型简介

`CogAgent-9B-2024122` 模型基于 [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b)
双语开源VLM基座模型，通过数据的采集与优化、多阶段训练与策略改进等方法，`CogAgent-9B-20241220` 在GUI
感知、推理预测准确性、动作空间完善性、任务的普适和泛化性上得到了大幅提升，能够接受中英文双语的屏幕截图和语言交互。
此版CogAgent模型已被应用于智谱AI的 [GLM-PC产品](https://cogagent.aminer.cn/home)
。我们希望这版模型的发布能够帮助到学术研究者们和开发者们，一起推进基于视觉语言模型的 GUI agent 的研究和应用。

### 能力展示

<div style="display: flex; flex-direction: column; width: 100%; align-items: center; margin-top: 20px;">
    <div style="text-align: center; margin-bottom: 20px; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/4d39fe6a-d460-427c-a930-b7cbe0d082f5" width="100%" height="auto" controls autoplay loop></video>
        <p>CogAgent 祝你圣诞快乐，让大模型自动为你的朋友们送上圣诞祝福吧。</p>
    </div>
    <div style="text-align: center; width: 100%; max-width: 600px; height: auto;">
        <video src="https://github.com/user-attachments/assets/87f00f97-1c4f-4152-b7c0-d145742cb910" width="100%" height="auto" controls autoplay loop></video>
        <p>想提个Issue,让 CogAgent帮你发邮件。</p>
    </div>
</div>

### 推理和微调成本

+ 模型在 `BF16` 精度下推理至少需要使用`29GB`显存。不建议使用 `INT4` 精度推理，性能损失较大。
+ 以上所有数据中的GPU指A100, H100 GPU，其他设备显存/内存需自行计算。
+ SFT过程中，本代码冻结`Vision Encoder`, Batch Size = 1, 使用`8 * A100` 进行微调，输入token(包含图像的`1600` tokens) 共计
  2048
  Tokens。本代码无法在`Vision Encoder`不冻结的情况下进行SFT微调。LORA过程中，不冻结`Vision Encoder`, Batch Size = 1, 使用
  `1 * A100` 进行微调，输入token(包含图像的`1600` tokens) 共计 2048 Tokens。在上述情况下，SFT微调需要每张GPU至少需要拥有
  `60GB`显存，8张GPU，LORA微调需要每张GPU至少需要拥有`70GB`显存，1张GPU，不可切割。
+ `昇腾设备` 未测试SFT微调。仅在`Atlas800训练服务器集群`上进行测试。具体推理代码需要根据`昇腾设备`下载链接中载入模型的方式进行修改。
+ 目前，暂时不支持`vLLM`框架进行推理。我们会尽快提交PR支持。
+ 在线体验链接不支持控制电脑，仅支持查看模型的推理结果。我们建议本地部署模型。

## 输入和输出

`cogagent-9b-20241220`是一个Agent类执行模型而非对话模型，不支持连续对话，但是但支持连续的执行历史。
这里展示了用户应该怎么整理自己的输入格式化的传入给模型。并获得模型规则的回复。

### 用户输入部分

1. `task` 字段

   用户输入的任务描述，类似文本格式的prompt，该输入可以指导`cogagent-9b-20241220`模型完成用户任务指令。请保证简洁明了。

2. `platform` 字段

   `cogagent-9b-20241220`支持在多个平台上执行可操作Agent功能, 我们支持的带有图形界面的操作系统有三个系统，
    - Windows 10，11，请使用 `WIN` 字段。
    - Mac 14，15，请使用 `Mac` 字段。
    - Android 13，14，15 以及其他GUI和UI操作方式几乎相同的安卓UI发行版，请使用 `Mobile` 字段。

   如果您使用的是其他系统，效果可能不佳，但可以尝试使用 `Mobile` 字段用于手机设备，`WIN` 字段用于Windows设备，`Mac`
   字段用于Mac设备。

3. `format` 字段

   用户希望`cogagent-9b-20241220`返回何种格式的数据, 这里有以下几种选项:
    - `Answer in Action-Operation-Sensitive format.`: 本仓库中demo默认使用的返回方式，返回模型的行为，对应的操作，以及对应的敏感程度。
    - `Answer in Status-Plan-Action-Operation format.`: 返回模型的装题，行为，以及相应的操作。
    - `Answer in Status-Action-Operation-Sensitive format.`: 返回模型的状态，行为，对应的操作，以及对应的敏感程度。
    - `Answer in Status-Action-Operation format.`: 返回模型的状态，行为。
    - `Answer in Action-Operation format.` 返回模型的行为，对应的操作。

4. `history` 字段

   拼接顺序和结果应该如下所示：
   ```
   query = f'{task}{history}{platform}{format}'
   ```

### 模型返回部分

1. 敏感操作: 包括 `<<敏感操作>> <<一般操作>>` 几种类型，只有要求返回`Sensitive`的时候返回。
2. `Plan`, `Status`, `Action` 字段: 用于描述模型的行为和操作。只有要求返回对应字段的时候返回，例如带有`Action`则返回
   `Action`字段内容。
3. 常规回答部分，这部分回答会在格式化回答之前，表示综述。
4. `Grounded Operation` 字段:
   用于描述模型的具体操作，包括操作的位置，类型，以及具体的操作内容。其中 `box` 代表执行区域的坐标，`element_type` 代表执行的元素类型，
   `element_info` 代表执行的元素描述。这些信息被一个 `操作指令` 操作所包裹。具体的动作空间请参考[这里](Action_space.md)。

### 例子

用户的任务是希望帮忙将所有邮件标记为已读，用户使用的是 Mac系统，希望返回的是Action-Operation-Sensitive格式。
正确拼接后的提示词应该为：

```
Task: 帮我将所有的邮件标注为已读
(Platform: Mac)
(Answer in Action-Operation-Sensitive format.)
```

接着，这里展现了不同格式要求下的返回结果:


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

### 注意事项

1. 该模型不是对话模型，不支持连续对话，请发送具体指令，并参考我们提供的历史拼接方式进行拼接。
2. 该模型必须要有图片传入，纯文字对话无法实现GUI Agent任务。
3. 该模型输出有严格的格式要求，请严格按照我们的要求进行解析。输出格式为 STR 格式，不支持输出JSON 格式。

## 运行模型

### 环境配置

请确保你已安装 **Python 3.10.16** 或者以上版本。并安装以下依赖:

```shell
pip install -r requirements.txt
```

运行一个本地的基于`transformers`的模型推理，你可以通过运行以下命令来运行模型:

```shell
python inference/cli_demo.py --model_dir THUDM/cogagent-9b-20241220 --platform "Mac" --max_length 4096 --top_k 1 --output_image_path ./results --format_key status_action_op_sensitive
```

这是一个命令行交互代码。你需要输入对应的图像路径。 如果模型返回的结果带有bbox，则会输出一张带有bbox的图片，表示需要在这个区域内执行操作，保存的图片为路径为
`output_image_path`中，图片名称为 `{你输入的图片名}_{对话轮次}.png` 。`format_key` 表示你希望通过模型通过哪种格式返回。
`platform` 字段则决定了你服务于哪种平台（比如`Mac`,则你上传的截图都必须是`Mac`系统的截图）。

如果你希望运行在线 web demo，这是一个需要连续上传图片进行交互的demo，模型将会返回对应的Bbox和执行类别。该代码与
`HuggingFace Space`
在线体验效果相同。

```shell
python inference/web_demo.py --host 0.0.0.0 --port 7860 --model_dir THUDM/cogagent-9b-20241220 --format_key status_action_op_sensitive --platform "Mac" --output_dir ./results
```

### 运行 Agent APP 示例

我们为开发者准备了一个基础的Demo APP，用于演示`cogagent-9b-20241220`模型的GUI能力，该Demo展示了如何在带有GPU的服务器上部署模型，
并在本地的电脑上运行`cogagent-9b-20241220`模型执行自动化GUI操作。

> 我们无法保证AI的行为的安全性，请在使用时谨慎操作。
>
> 本示例仅供学术参考，我们不承担由本示例引起的任何法责任。

如果你对该 APP 感兴趣，欢迎查看[文档](app/README.md)

### 微调模型

如果你对微调`cogagent-9b-20241220`模型感兴趣，欢迎查看[这里](finetune/README.md)。

## 先前的工作

在2023年11月，我们发布了CogAgent的第一代模型，现在，你可以在 [CogVLM&CogAgent官方仓库](https://github.com/THUDM/CogVLM)
找到相关代码和权重地址。

<div align="center">
    <img src=assets/cogagent_function_cn.jpg width=70% />
</div>

<table>
  <tr>
    <td>
      <h2> CogVLM </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2311.03079">CogVLM: Visual Expert for Pretrained Language Models</a></p>
      <p><b>CogVLM</b> 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B拥有100亿的视觉参数和70亿的语言参数，支持490*490分辨率的图像理解和多轮对话。</p>
      <p><b>CogVLM-17B 17B在10个经典的跨模态基准测试中取得了最先进的性能</b>包括NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA 和 TDIUC 基准测试。</p>
    </td>
    <td>
      <h2> CogAgent </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2312.08914">CogAgent: A Visual Language Model for GUI Agents </a></p>
      <p><b>CogAgent</b> 是一个基于CogVLM改进的开源视觉语言模型。CogAgent-18B拥有110亿的视觉参数和70亿的语言参数, <b>支持1120*1120分辨率的图像理解。在CogVLM的能力之上，它进一步拥有了GUI图像Agent的能力。</b></p>
      <p> <b>CogAgent-18B 在9个经典的跨模态基准测试中实现了最先进的通用性能，</b>包括 VQAv2, OK-VQ, TextVQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, 和 POPE 测试基准。它在包括AITW和Mind2Web在内的GUI操作数据集上显著超越了现有的模型。</p>
    </td>
  </tr>
</table>

## 协议

- 本 github 仓库代码的使用 [Apache2.0 LICENSE](LICENSE)。

- 模型权重的使用请遵循 [Model License](MODEL_LICENSE)。
