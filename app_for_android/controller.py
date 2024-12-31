import os
import re
import time
import subprocess
from PIL import Image


def get_location(adb_path):
    # 返回经纬度
    command = adb_path + " shell dumpsys location"
    result = subprocess.run(command, capture_output=True, text=True, shell=True).stdout

    # 使用正则，查找经纬度
    # last location=Location[network 40.003831,116.329476 hAcc=15.0 et=+1d1h11m29s41ms {Bundle[{indoor=0, provider=wifi, source=2}]}]
    pattern = r"last location=Location\[network(.*?)hAcc="
    match = re.search(pattern, result)
    if match:
        location = match.group(1).strip()
        return location

    return ""


def get_phone_time(adb_path):
    command = adb_path + " shell date +%Y-%m-%d_%H:%M:%S"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip()


def set_default_input_method(adb_path):
    # 设置默认输入法为 ADB Keyboard
    command = adb_path + " shell ime set com.android.adbkeyboard/.AdbIME"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 设置默认输入法失败")
        print(process.stderr)
    else:
        print("默认输入法已设置为 ADB Keyboard")


def check_devices(adb_path):
    command = adb_path + " devices"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 检查设备失败")
        print(process.stderr)
    else:
        print("设备已检查")


def get_screen_size(adb_path):
    command = adb_path + " shell wm size"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 获取屏幕大小失败")
        print(process.stderr)
    else:
        # print("屏幕大小已获取")
        # Physical size: 1080x1920
        width, height = process.stdout.split()[-1].split("x")
        return int(width), int(height)


def get_screenshot(adb_path, print_flag=True, round_num=None):
    if print_flag:
        print("正在获取截图....")
    # 删除旧的截图
    command = adb_path + " shell rm /sdcard/screenshot.png"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 删除截图失败")
        print(process.stderr)
    time.sleep(0.5)
    # 获取新的截图
    command = adb_path + " shell screencap -p /sdcard/screenshot.png"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 截图失败")
        print(process.stderr)
    time.sleep(0.5)
    # 下载截图
    command = adb_path + " pull /sdcard/screenshot.png ./screenshot"
    process = subprocess.run(command, capture_output=True, text=True, shell=True)
    if process.returncode != 0:
        print("Error: 拉取截图失败")
        print(process.stderr)
    time.sleep(0.5)
    image_path = "./screenshot/screenshot.png"
    if round_num is not None:
        save_path = f"./screenshot/screenshot_{round_num}.jpg"
    else:
        save_path = "./screenshot/screenshot.jpg"
    image = Image.open(image_path)
    image.convert("RGB").save(save_path, "JPEG")
    os.remove(image_path)

    return save_path


#####################


def tap1(adb_path, params):
    """
    Meta-operation: CLICK
    CLICK: 使用swipe在框的中心位置模拟Tap。
    """
    x, y = params["box"]
    x1, y1 = x, y
    x2, y2 = x * 1.01, y * 1.01
    command = adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 200"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def tap2(adb_path, params):
    """
    Meta-operation: CLICK
    CLICK: 在框的中心位置模拟Tap。
    """
    x, y = params["box"]
    command = adb_path + f" shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def double_tap(adb_path, params):
    """
    Meta-operation: DOUBLE CLICK
    DOUBLE CLICK: 在框的中心位置模拟双击。
    """
    x, y = params["box"]

    command = adb_path + f" shell input tap {x} {y}"
    command = f"{command} && {command}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type_input(adb_path, params):
    text = params["text"]
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == " ":
            command = adb_path + f" shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == "_":
            command = adb_path + f" shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif "a" <= char <= "z" or "A" <= char <= "Z" or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in "-.,!?@'°/:;()":
            command = adb_path + f' shell input text "{char}"'
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f' shell am broadcast -a ADB_INPUT_TEXT --es msg "{char}"'
            subprocess.run(command, capture_output=True, text=True, shell=True)

    # 回车 KEYCODE_ENTER
    command = adb_path + f" shell input keyevent 66"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def scroll_down(adb_path, params):
    """
    Meta-operation: SCROLL DOWN
    SCROLL DOWN: 在框的中心位置模拟向下滚动。
    """
    x, y = params["box"]
    width, height = params["screen_size"]
    command = adb_path + f" shell input swipe {x} {y} {x} {height*0.9} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def scroll_up(adb_path, params):
    """
    Meta-operation: SCROLL UP
    SCROLL UP: 在框的中心位置模拟向上滚动。
    """
    x, y = params["box"]
    width, height = params["screen_size"]
    command = adb_path + f" shell input swipe {x} {y} {x} {min(height*0.1, y*0.5)} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def scroll_right(adb_path, params):
    """
    Meta-operation: SCROLL RIGHT
    SCROLL RIGHT: 在框的中心位置模拟向右滚动。
    """
    x, y = params["box"]
    width, height = params["screen_size"]
    command = adb_path + f" shell input swipe {width*0.1} {height/2} {width*0.9} {height/2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def scroll_left(adb_path, params):
    """
    Meta-operation: SCROLL RIGHT
    SCROLL RIGHT: 在框的中心位置模拟向左滚动。
    """
    x, y = params["box"]
    width, height = params["screen_size"]
    command = adb_path + f" shell input swipe {width*0.9} {height/2} {width*0.1} {height/2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def key_press(adb_path, params):
    """
    Meta-operation: KEY_PRESS
    TYPE: Press a special key on the keyboard. eg: KEY_PRESS(key='Return').
    """
    key = params["key"]
    command = adb_path + f" shell input keyevent {KEY_CODE[key]}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def launch(adb_path, params):
    # TODO 需要测试
    """
    Meta-operation: LAUNCH
    LAUNCH: Launch an app by package name. eg: LAUNCH(app='com.example.app').
    """
    app = params["app"]
    # app = params["app"][1:-1] WHY?
    if app in package_dict.keys():
        package_activity = package_dict[app]
        command = adb_path + f" shell am start -n {package_activity}"
        subprocess.run(command, capture_output=True, text=True, shell=True)
    else:
        print("App not found in package_dict.")
        return


def end(adb_path, params):
    print("Workflow Completed!")


#####################


def back(adb_path):
    command = adb_path + f" shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def home(adb_path):
    command = (
        adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    )
    subprocess.run(command, capture_output=True, text=True, shell=True)


# https://blog.csdn.net/jlminghui/article/details/39268419
# TODO
KEY_CODE = {}

# TODO: Support other META_PARAMETER and other META_OPERATION
META_PARAMETER = {
    # Meta-operations need to contain keywords
    "CLICK": ["box"],
    "DOUBLE_CLICK": ["box"],
    # "RIGHT_CLICK": ["box"],
    "TYPE": ["box", "text"],
    # "HOVER": ["box"],
    "SCROLL_DOWN": ["box"],
    "SCROLL_UP": ["box"],
    "SCROLL_RIGHT": ["box"],
    "SCROLL_LEFT": ["box"],
    # "KEY_PRESS": ["key"],
    "LAUNCH": ["app"],
    # "QUOTE_TEXT": ["box"],
    # "QUOTE_CLIPBOARD": ["output"],
    # "TEXT_FORMAT": ["input"],
    # "LLM": ["prompt"],
    "END": [""],
}

META_OPERATION = {
    # Defining meta-operation functions
    "CLICK": tap1,
    "DOUBLE_CLICK": double_tap,
    # "RIGHT_CLICK": right_click, # Mobile没有右键
    "TYPE": type_input,
    # "HOVER": hover, # Mobile没有悬停
    "SCROLL_DOWN": scroll_down,
    "SCROLL_UP": scroll_up,
    "SCROLL_RIGHT": scroll_right,  # Mobile 需要右划
    "SCROLL_LEFT": scroll_left,  # Mobile 需要左划
    # "KEY_PRESS": key_press,
    "LAUNCH": launch,
    # "QUOTE_TEXT": ["box"],
    # "QUOTE_CLIPBOARD": ["output"],
    # "TEXT_FORMAT": ["input"],
    # "LLM": ["prompt"],
    "END": end,
}

# 包名/Activity 和 App名的对应关系
# TODO: 需要更新 & 不同手机不一样
package_dict = {
    "抖音": "com.ss.android.ugc.aweme/com.ss.android.ugc.aweme.splash.SplashActivity",
    "饿了么": "me.ele/me.ele.Launcher",
    "微博": "com.sina.weibo/com.sina.weibo.SplashActivity",
    "微信": "com.tencent.mm/com.tencent.mm.ui.LauncherUI",
    "小红书": "com.xingin.xhs/com.xingin.xhs.index.v2.IndexActivityV2",
    "桌面": "com.miui.home/com.miui.home.launcher.Launcher",
    "天气": "com.miui.weather2/com.miui.weather2.ActivityWeatherMain",
    "京东": "com.jingdong.app.mall/com.jingdong.app.mall.MainFrameActivity",
    "时钟": "com.android.deskclock/com.android.deskclock.DeskClockTabActivity",
    "计算器": "com.miui.securitycenter/com.miui.permcenter.permissions.SystemAppPermissionDialogActivity",
    "拼多多": "com.xunmeng.pinduoduo/com.xunmeng.pinduoduo.ui.activity.MainFrameActivity",
    "腾讯体育": "com.tencent.qqsports/com.tencent.qqsports.ui.SplashActivity",
    "支付宝": "com.eg.android.AlipayGphone/com.eg.android.AlipayGphone.AlipayLogin",
    "相册": "com.miui.gallery/com.miui.gallery.activity.HomePageActivity",
    "淘宝": "com.taobao.taobao/com.taobao.tao.welcome.Welcome",
    "番茄小说": "com.dragon.read/com.dragon.read.pages.splash.SplashActivity",
    "美团": "com.sankuai.meituan/com.meituan.android.pt.homepage.activity.MainActivity                                                                                          ",
    "地铁": "com.ruubypay.bjmetro/com.ruubypay.bjmetro.main.SplashADMasterActivity",
    "哈啰单车": "com.jingyao.easybike/com.hellobike.atlas.business.portal.PortalActivity",
    "高德地图": "com.autonavi.minimap/com.autonavi.map.activity.SplashActivity",
    "腾讯会议": "com.tencent.wemeet.app/com.tencent.wemeet.app.StartupActivity",
    "头条": "com.ss.android.article.news/com.ss.android.article.news.activity.MainActivity",
}


def convert_to_meta_operation(Grounded_Operation, adb_path):
    """
    元操作转换

    Parameters:
    - Grounded_Operation: Dict[str, Any] - 元操作
        {
            box: [425, 570, 573, 638],
            element_info: '[android.widget.ImageView]',
            operation: CLICK,
            app: '饿了么',
            url: 'None',
        }
    """
    detailed_operation = {}
    op = Grounded_Operation["operation"]
    if op in META_PARAMETER:
        #  例如 CLICK, TYPE, SLIDE, BACK,
        detailed_operation["meta"] = op
        for value in META_PARAMETER[op]:
            if value in Grounded_Operation:
                if value == "box":
                    # number = (left, top, width, height)
                    numbers = Grounded_Operation["box"]
                    box = [num / 1000 for num in numbers]
                    # box = (left/1000, top/1000, width/1000, height/1000)
                    width, height = get_screen_size(adb_path)
                    # x_min, y_min, x_max, y_max = (left, top, right, down)
                    # x_min, y_min, x_max, y_max = [
                    #     int(coord * width) if i % 2 == 0 else int(coord * height)
                    #     for i, coord in enumerate(box)
                    # ]
                    x_min = int(box[0] * width)
                    y_min = int(box[1] * height)
                    x_max = int(box[2] * width)
                    y_max = int(box[3] * height)
                    x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
                    detailed_operation["box"] = (x, y)  # 中心点
                    detailed_operation["screen_size"] = (width, height)  # 屏幕大小
                else:
                    detailed_operation[value] = Grounded_Operation[value][1:-1]
        return detailed_operation
    else:
        raise f"Wrong operation or operation not registered! {op=}"


def mobile_agent(Grounded_Operation, adb_path):
    """
    执行操作

    Parameters:
    - Grounded_Operation: Dict[str, Any] - 元操作
        {
            box: [425, 570, 573, 638],
            element_info: '[android.widget.ImageView]',
            operation: CLICK,
            app: '饿了么',
            url: 'None',
        }

    """
    # 提取 元操作
    detailed_operation = convert_to_meta_operation(Grounded_Operation, adb_path)
    # 执行 元操作
    META_OPERATION[detailed_operation["meta"]](adb_path, detailed_operation)
    time.sleep(2)
    return detailed_operation["meta"]


if __name__ == "__main__":
    pass
