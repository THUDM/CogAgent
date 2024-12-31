## 部署 Agent for Android Demo


### 🔧运行服务端

1. 在本步骤之前，清确保已经安装了本项目的首页的`requirements.txt`的全部依赖。这些依赖能保证服务端正常运行。
2. 在远程服务器拉起服务(app目录内)。

```shell
cd app
python openai_demo.py --model_path THUDM/cogagent-9b-20241220 --host 0.0.0.0 --port 7870
```

这将在服务器拉起一个模仿`OpenAI`接口格式的服务端，默认端口部署在 http://0.0.0.0:7870 。

### 🔧安装环境(用户端)

#### 安装用户端的依赖。
```shell
pip install -r requirements.txt
```


#### 准备通过ADB连接你的移动设备

1. 下载 [Android Debug Bridge](https://developer.android.com/tools/releases/platform-tools?hl=en)（ADB）。
2. 在你的移动设备上开启“USB调试”或“ADB调试”，它通常需要打开开发者选项并在其中开启。如果是HyperOS系统需要同时打开 "[USB调试(安全设置)](https://github.com/user-attachments/assets/05658b3b-4e00-43f0-87be-400f0ef47736)"。
3. 通过数据线连接移动设备和电脑，在手机的连接选项中选择“传输文件”。
4. 用下面的命令来测试你的连接是否成功: ```/path/to/adb devices```。如果输出的结果显示你的设备列表不为空，则说明连接成功。
5. 如果你是用的是MacOS或者Linux，请先为 ADB 开启权限: ```sudo chmod +x /path/to/adb```。
6.  ```/path/to/adb```在Windows电脑上将是```xx/xx/adb.exe```的文件格式，而在MacOS或者Linux则是```xx/xx/adb```的文件格式。

#### 在你的移动设备上安装 ADB 键盘
1. 下载 ADB 键盘的 [apk](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk)  安装包。
2. 在设备上点击该 apk 来安装。
3. 在系统设置中将默认输入法切换为 “ADB Keyboard”。`adb shell ime set com.android.adbkeyboard/.AdbIME`


### 🔧运行客户端

运行客户端，请确定以下信息:

0. 使用`adb kill-server` 删除假的emulator 或者 把 ADB 添加 ` -s xxxxx `
1. 在 ```client_for_android.py``` 的22行起编辑你的设置， 并且输入你的 ADB 路径和 API_url。
2. 请确保服务端已经正常运行,并确认服务器端和本地已经通过内网穿透等技术联通。
    请确保服务端可以从外网访问，或者通过内网穿透的方式允许自己的本地访问。在我们的代码中，服务端穿透到本地的端口为`7866`，所以环境变量应该设置为 `http://localhost:7866/v1` 。
    本 Demo没有设置 API, 因此`api_key`参数设置为`EMPTY`。

```shell
python client_for_android.py 
```

通过上述命令，你将能在本地运行客户端，连接到服务端，并且使用`cogagent-9b-20241220`模型。



> 我们无法保证AI的行为的安全性，请在使用时谨慎操作。本示例仅供学术参考，我们不承担由本示例引起的任何法责任。
>
> 模型运行中，你可以随时按下`stop`强制停止模型当前的操作。
>
> 如果你认为当前模型执行正常，没有风险，请不要触碰电脑，模型需要根据实时的电脑截图来确定点击坐标，这是模型正常运行的必要条件。



## 📦相关项目

* [Mobile-Agent: The Powerful Mobile Device Operation Assistant Family](https://github.com/X-PLUG/MobileAgent)
