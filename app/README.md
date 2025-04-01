## 部署 Agent Demo

### 设备检查

本 Demo 的测试环境系统环境如下:

```
macOS Sequoia: Version 15.0.1 (24A348)
Memory: 16GB
Python Version: 3.13.1 / 3.10.16
```

对于其他与上述环境不同的配置 (例如Windows操作系统，Linux桌面版操作系统)，我们未进行测试，其依赖库`pyautogui`支持Windows操作系统，开发者可自行尝试和丰富Demo。


### 安装环境(用户端)

在本步骤之前，清确保已经安装了本项目的首页的`requirements.txt`的全部依赖。这些依赖能保证服务端正常运行。接着，按照下面的步骤安装用户端的依赖。

1. 请在mac系统环境中安装`tkinter`库。具体安装方式可以参考下面代码

```shell
brew install python-tk
pip install -r requirements.txt
```

为了验证系统 python 是否已经正常安装`tkinter`库，可以在终端中输入下面代码

```shell
/opt/homebrew/bin/python3 -m tkinter
```

正常应该返回

```
2024-12-14 15:29:04.041 Python[7161:122540731] +[IMKClient subclass]: chose IMKClient_Legacy
2024-12-14 15:29:04.041 Python[7161:122540731] +[IMKInputSession subclass]: chose IMKInputSession_Legacy
```

2. 创建虚拟环境，请不要使用`conda`,`virtualenv`等工具，因为这些工具会导致`tkinter`库无法正常使用
   请将下面代码复制到终端中执行，并替换`/Users/zr/Code/CogAgent/venv`为你的实际路径

```shell
/opt/homebrew/bin/python3 -m venv --copies /Users/zr/Code/CogAgent/venv
```

你需要验证虚拟环境是否创建成功，可以在终端中输入下面代码

```shell
/Users/zr/Code/CogAgent/venv/bin/python3 -m tkinter
```

请确保在本文件夹下创建一个`caches`文件夹，用于保存模型执行中的照片。

```shell
mkdir caches
```

3. 确保你的电脑设备给予了足够权限，一般来说，你执行代码的软件需要赋予截图，录屏以及模拟键盘鼠标操作的权限。我们展现了如何在Mac设备中开启这些权限。

| 开启录屏权限                    | 开启键盘鼠标操作权限                |
|---------------------------|---------------------------|
| ![1](../assets/app_1.png) | ![2](../assets/app_2.png) |

在这里，我们使用 `Pycharm` 来运行 `client.py` 程序。因此，需要给予软件 `Pycharm` 以及 `terminal`权限，如果你仅仅在终端执行，仅需要给予
`terminal`权限。`VSCode` 等其他IDE操作方式同理。


### 运行服务端

在远程服务器拉起服务

```shell
python openai_demo.py --model_path THUDM/cogagent-9b-20241220 --host 0.0.0.0 --port 7870
```

或者使用vllm启动远程服务

```shell
python vllm_openai_demo.py --model_path THUDM/cogagent-9b-20241220 --host 0.0.0.0 --port 7870
```

这将在服务器拉起一个模仿`OpenAI`接口格式的服务端，默认端口部署在 http://0.0.0.0:7870 。

### 运行客户端

运行客户端，请确定以下信息:

- 请确保服务端已经正常运行,并确认服务器端和本地已经通过内网穿透等技术联通。
- 请确保服务端可以从外网访问，或者通过内网穿透的方式允许自己的本地访问。在我们的代码中，服务端穿透到本地的端口为`7870`
  ，所以环境变量应该设置为 http://127.0.0.1:7870/v1 。
- 本 Demo没有设置 API, 因此`api_key`参数设置为`EMPTY`。

```shell
python client.py --api_key EMPTY --base_url http://127.0.0.1:7870/v1  --client_name 127.0.0.1 --client_port 7860 --model CogAgent
```

通过上述命令，你将能在本地运行客户端，连接到服务端，并且使用`cogagent-9b-20241220`模型。
下图展现了正常启动APP并让模型接管电脑到截图（图中右下角小火箭是APP）。

![img.png](../assets/app_gradio.png)

> 我们无法保证AI的行为的安全性，请在使用时谨慎操作。本示例仅供学术参考，我们不承担由本示例引起的任何法责任。
>
> 模型运行中，你可以随时按下`stop`强制停止模型当前的操作。
>
> 如果你认为当前模型执行正常，没有风险，请不要触碰电脑，模型需要根据实时的电脑截图来确定点击坐标，这是模型正常运行的必要条件。



