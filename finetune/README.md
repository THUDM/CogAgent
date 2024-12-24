# CogAgent 模型微调

Read this in [English](README_en.md)

本 demo 中，你将体验到如何微调 CogAgent 开源模型。 请严格按照文档的步骤进行操作，以避免不必要的错误。

## 多轮微调格式

多轮微调示例采用 CogAgent 对话格式约定，对不同角色添加不同 `loss_mask` 从而在一遍计算中为多轮回复计算 `loss`。

对于数据文件，样例采用如下格式：

对于 cogagent-9b-20241220 模型，您应该按照以下格式整理数据, 保证每一条数据都有图片并且`user`的History steps部分包括之前已有的全部操作。

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Task: 通过启动台打开系统设置，导航到“网络”部分，将DNS设置手动IP地址为8.8.4.4。\n(Platform: Mac)\n(Answer in Action-Operation-Sensitive format.)\nHistory steps: \n0. CLICK(box=[[7,6,27,31]], element_info='苹果标志')\t点击屏幕左上角的“苹果”图标，以便打开系统设置。\n1. CLICK(box=[[3,77,161,99]], element_info='系统设置…')\t点击屏幕左上角菜单栏中的“系统设置…”选项，以打开系统设置界面。\n2. CLICK(box=[[7,478,125,512]], element_info='网络')\t点击左侧菜单中的“网络”选项，进入网络设置界面。\n3. CLICK(box=[[155,217,459,270]], element_info='Wi-Fi, 已连接')\t点击屏幕中间偏上的“Wi-Fi”选项，进入Wi-Fi设置界面。\n4. CLICK(box=[[405,280,447,296]], element_info='详细信息…')\t点击Wi-Fi名称旁边的“详细信息…”按钮，以查看当前Wi-Fi的详细信息。\n5. CLICK(box=[[27,344,45,362]], element_info='DNS')\t点击左侧菜单中的“DNS”选项，进入DNS设置界面。\n6. CLICK(box=[[166,308,442,399]], element_info='8.8.8.8')\t点击DNS服务器地址输入框，准备输入新的DNS服务器地址。\n7. TYPE(box=[[163,308,443,322]], text='8.8.4.4', element_info='ip地址输入框')\t在DNS服务器地址输入框中输入8.8.4.4，以设置手动IP地址。\n8. CLICK(box=[[410,636,440,651]], element_info='好')\t点击界面右下角的“好”按钮，以确认并保存DNS设置的更改。\n",
      "image": "images/0000000000482.png"
    },
    {
      "role": "assistant",
      "content": "Action: 已经完成了“通过启动台打开系统设置，导航到‘网络’部分，将DNS设置手动IP地址为8.8.4.4”这一任务，执行结束。\nGrounded Operation:END()\n<<END>>"
    }
  ]
}
```

## 配置文件

微调配置文件位于 `config` 目录下，包括以下文件：

1. `ds_zereo_2 / ds_zereo_3.json`: deepspeed 配置文件。

2. `lora.yaml / sft.yaml`: 模型不同方式的配置文件，包括模型参数、优化器参数、训练参数等。 部分重要参数解释如下：
    + data_config 部分
        + train_file: 训练数据集的文件路径。
        + val_file: 验证数据集的文件路径。
        + test_file: 测试数据集的文件路径。
        + num_proc: 在加载数据时使用的进程数量。
    + freezeV: 是否冻结vision部分参数。
    + max_input_length: 输入序列的最大长度, 请注意，在模型实际的推理中，还会固定加入`1600` token 的图像编码结果。
    + max_output_length: 输出序列的最大长度。
    + training_args 部分
        + output_dir: 用于保存模型和其他输出的目录。
        + max_steps: 训练的最大步数。
        + per_device_train_batch_size: 每个设备（如 GPU）的训练批次大小。
        + dataloader_num_workers: 加载数据时使用的工作线程数量。
        + remove_unused_columns: 是否移除数据中未使用的列。
        + save_strategy: 模型保存策略（例如，每隔多少步保存一次）。
        + save_steps: 每隔多少步保存一次模型。
        + log_level: 日志级别（如 info）。
        + logging_strategy: 日志记录策略。
        + logging_steps: 每隔多少步记录一次日志。
        + per_device_eval_batch_size: 每个设备的评估批次大小。
        + evaluation_strategy: 评估策略（例如，每隔多少步进行一次评估）。
        + eval_steps: 每隔多少步进行一次评估。
        + predict_with_generate: 是否使用生成模式进行预测。
    + generation_config 部分
        + max_new_tokens: 生成的最大新 token 数量。
    + peft_config 部分
        + peft_type: 使用的参数有效调整类型 (支持 LORA 和 PREFIX_TUNING)。
        + task_type: 任务类型，这里是因果语言模型 (不要改动)。
    + Lora 参数：
        + r: LoRA 的秩。
        + lora_alpha: LoRA 的缩放因子。
        + lora_dropout: 在 LoRA 层使用的 dropout 概率。

## 开始微调

通过以下代码执行 **单机多卡/多机多卡** 运行，这是使用 `deepspeed` 作为加速方案的，您需要安装 `deepspeed`。
CogAgent1.5数据集由您自行准备，接着，按照此命令运行：

```shell
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/CogAgentData/  THUDM/cogagent-2-9b  configs/sft.yaml
```

通过以下代码执行 **单机单卡** 运行。

```shell
python finetune.py  data/CogAgentData/  THUDM/cogagent-9b-20241220  configs/lora.yaml
```

## 从保存点进行微调

如果按照上述方式进行训练，每次微调都会从头开始，如果你想从训练一半的模型开始微调，你可以加入第四个参数，这个参数有两种传入方式:

1. `yes`, 自动从最后一个保存的 Checkpoint开始训练
2. `XX`, 断点号数字 例 `600` 则从序号600 Checkpoint开始训练

例如，这就是一个从最后一个保存点继续微调的示例代码

```shell
python finetune.py  data/CogAgentData/  THUDM/cogagent-9b-20241220  configs/lora.yaml yes
```

## 采用华为昇腾计算计算设备进行微调

如果你需要使用`Ascend NPU`设备，例如`ATLAS 300 A2`，你需要解除注释:

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

之后就能正常运行微调程序。
