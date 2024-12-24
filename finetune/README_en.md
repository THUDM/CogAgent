# CogAgent model fine-tuning

In this demo, you will experience how to fine-tune the CogAgent open source model. Please strictly follow the steps in
the document to avoid unnecessary errors.

## Multiple rounds of fine-tuning format

The multi-round fine-tuning example uses the CogAgent dialogue format convention, adding different `loss_mask` to
different roles to calculate the `loss` for multiple rounds of replies in one calculation.

For the data file, the sample uses the following format:

For the cogagent-9b-20241220 model, you should organize the data in the following format, ensuring that each data entry has an image and the History steps section of `user` includes all previous operations.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Task: Open the system settings through the launch pad, navigate to the "Network" section, and set the DNS manual IP address to 8.8.4.4.\n(Platform: Mac)\n(Answer in Action-Operation-Sensitive format.)\nHistory steps: \n0. CLICK(box=[[7,6,27,31]], element_info='Apple logo')\tClick the "Apple" icon in the upper left corner of the screen to open the system settings.\n1. CLICK(box=[[3,77,161,99]], element_info='System settings…')\tClick the "System settings…" option in the menu bar in the upper left corner of the screen to open the system settings interface.\n2. CLICK(box=[[7,478,125,512]], element_info='Network')\tClick the "Network" option in the left menu to enter the network settings interface. \n3. CLICK(box=[[155,217,459,270]], element_info='Wi-Fi, Connected')\tClick the "Wi-Fi" option in the upper middle of the screen to enter the Wi-Fi settings interface. \n4. CLICK(box=[[405,280,447,296]], element_info='Details...')\tClick the "Details..." button next to the Wi-Fi name to view detailed information about the current Wi-Fi. \n5. CLICK(box=[[27,344,45,362]], element_info='DNS')\tClick the "DNS" option in the left menu to enter the DNS settings interface. \n6. CLICK(box=[[166,308,442,399]], element_info='8.8.8.8')\tClick the DNS server address input box to enter a new DNS server address. \n7. TYPE(box=[[163,308,443,322]], text='8.8.4.4', element_info='ip address input box')\tEnter 8.8.4.4 in the DNS server>Address input box to set a manual IP address. \n8. CLICK(box=[[410,636,440,651]], element_info='OK')\tClick the "OK" button in the lower right corner of the interface to confirm and save the changes to the DNS settings. \n",
      "image": "images/0000000000482.png"
    },
    {
      "role": "assistant",
      "content": "Action: Left click\nGrounded Operation:CLICK(box=[[150,911,180,975]], element_info='lower')\n<<END>>"
    },
  ]
}
```

## Configuration file

The fine-tuning configuration files are located in the `config` directory and include the following files:
1. `ds_zereo_2 / ds_zereo_3.json`: DeepSpeed configuration files.
2. `lora.yaml / sft.yam`l: Configuration files for different model fine-tuning methods, including model parameters,
optimizer parameters, training parameters, and more. Some important parameter explanations are as follows:

   + Data Configuration Section
     + train_file: Path to the training dataset file.
     + val_file: Path to the validation dataset file.
     + test_file: Path to the test dataset file.
     + num_proc: Number of processes to use when loading data.

   + freezeV: Whether to freeze the vision model parameters.
   + max_input_length: Maximum input sequence length. Note that an additional 1600 tokens for image encoding results will
   be fixed during the model’s actual inference.
   + max_output_length: Maximum output sequence length.

   + Training Configuration Section
       + output_dir: Directory to save the model and other outputs.
       + max_steps: Maximum number of training steps.
       + per_device_train_batch_size: Training batch size per device (e.g., GPU).
       + dataloader_num_workers: Number of workers to use when loading data.
       + remove_unused_columns: Whether to remove unused columns from the data.
       + save_strategy: Model saving strategy (e.g., save every N steps).
       + save_steps: Number of steps after which the model will be saved.
       + log_level: Logging level (e.g., info).
       + logging_strategy: Logging strategy.
       + logging_steps: Number of steps after which logs will be recorded.
       + per_device_eval_batch_size: Evaluation batch size per device.
       + evaluation_strategy: Evaluation strategy (e.g., evaluate every N steps).
       + eval_steps: Number of steps after which evaluation will occur.
       + predict_with_generate: Whether to use generation mode for prediction.

   + Generation Configuration Section
     + max_new_tokens: Maximum number of new tokens to generate.

   + PEFT (Parameter Efficient Fine-Tuning) Configuration Section
     + peft_type: Type of parameter-efficient fine-tuning used (supports LORA and PREFIX_TUNING).
     + task_type: Task type, which is causal language modeling (do not modify).

   + LoRA Parameters:
     + r: Rank of the LoRA.
     + lora_alpha: Scaling factor for LoRA.
     + lora_dropout: Dropout probability used in the LoRA layer.

## Start fine-tuning

Execute **single machine multi-card/multi-machine multi-card** run through the following code, which uses `deepspeed` as
the acceleration solution, and you need to install `deepspeed`.
The cogagent-9b-20241220 model dataset is prepared by yourself, then run the following command:

```shell
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/CogAgentData  THUDM/cogagent-9b-20241220  configs/sft.yaml
```

Execute **single machine single card** run through the following code.

```shell
python finetune.py  data/CogAgentData/  THUDM/cogagent-9b-20241220  configs/lora.yaml
```

## Fine-tune from a saved point

If you train as described above, each fine-tuning will start from the beginning. If you want to fine-tune from a
half-trained model, you can add a fourth parameter, which can be passed in two ways:

1. `yes`, automatically start training from the last saved Checkpoint

2. `XX`, breakpoint number, for example `600`, start training from Checkpoint 600

For example, this is an example code to continue fine-tuning from the last saved point

```shell
python finetune.py data/CogAgent/ THUDM/cogagent-9b-20241220 configs/lora.yaml yes
```

## Fine-Tuning with Huawei Ascend Computing Devices

If you need to use `Ascend NPU` devices, such as `ATLAS 300 A2`, you should uncomment the following lines:

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

After that, you can run the fine-tuning program as expected.
