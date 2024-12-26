"""
This file registers all meta-operations that CogAgent1.5-9B Model can perform,
along with the required keywords that must be included for each meta-operation.

key: value
meta-operation: keyword
"""

import pyautogui
import pyperclip
import time
import os
import platform

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1

# TODO: Support other META_PARAMETER and other META_OPERATION
META_PARAMETER = {
    # Meta-operations need to contain keywords
    "CLICK": ["box"],
    "DOUBLE_CLICK": ["box"],
    "RIGHT_CLICK": ["box"],
    "TYPE": ["box", "text"],
    "HOVER": ["box"],
    "SCROLL_DOWN": ["box"],
    "SCROLL_UP": ["box"],
    # "SCROLL_RIGHT": ["box"],
    # "SCROLL_LEFT": ["box"],
    "KEY_PRESS": ["key"],
    "LAUNCH": ["app"],
    # "QUOTE_TEXT": ["box"],
    # "QUOTE_CLIPBOARD": ["output"],
    # "TEXT_FORMAT": ["input"],
    # "LLM": ["prompt"],
    "END": [""],
}


def identify_os():
    os_detail = platform.platform()
    if "mac" in os_detail.lower():
        return "Mac"
    elif "windows" in os_detail.lower():
        return "Win"
    else:
        raise ValueError(
            f"This {os_detail} operating system is not currently supported!"
        )


def paste(text):
    pyperclip.copy(text)
    time.sleep(1)
    if identify_os() == "Mac":
        with pyautogui.hold("command"):
            pyautogui.press("v")
    elif identify_os() == "Win":
        with pyautogui.hold("ctrl"):
            pyautogui.press("v")


def click(params):
    """
    Meta-operation: CLICK
    CLICK: Simulate a left-click at the center position of the box.
    """
    pyautogui.doubleClick(params["box"])


def double_click(params):
    """
    Meta-operation: DOUBLE_CLICK
    DOUBLE_CLICK: Simulate a double-click the center position of the box.
    """
    pyautogui.doubleClick(params["box"])


def right_click(params):
    """
    Meta-operation: RIGHT_CLICK
    RIGHT_CLICK: Simulate a right-click at the center position of the box.
    """
    pyautogui.rightClick(params["box"])


def type_input(params):
    """
    Meta-operation: TYPE
    TYPE: At the center position of the box, simulate keyboard input to enter text.
    """
    paste(params["text"])
    pyautogui.press("Return")


def hover(params):
    """
    Meta-operation: HOVER
    HOVER: Move the mouse to the center position of the box.
    """
    pyautogui.moveTo(params["box"])


def scroll_down(params):
    """
    Meta-operation: SCROLL_DOWN
    SCROLL_DOWN: Move the mouse to the center position of the box, then scroll the screen downward.
    """
    pyautogui.moveTo(params["box"])
    pyautogui.scroll(-10)


def scroll_up(params):
    """
    Meta-operation: SCROLL_UP
    SCROLL_UP: Move the mouse to the center position of the box, then scroll the screen up.
    """
    pyautogui.moveTo(params["box"])
    pyautogui.scroll(10)


def key_press(params):
    """
    Meta-operation: KEY_PRESS
    TYPE: Press a special key on the keyboard. eg: KEY_PRESS(key='Return').
    """
    pyautogui.press(params["key"])


def end(params):
    print("Workflow Completed!")


def launch(params):
    system_app_dir = "/System/Applications"  # For Mac
    applications_dir = "/Applications"  # For Mac
    applications = [app for app in os.listdir(applications_dir) if app.endswith(".app")]
    system_apps = [app for app in os.listdir(system_app_dir) if app.endswith(".app")]
    all_apps = applications + system_apps
    for app in all_apps:
        if params["app"][1:-1] in app:
            app_dir = applications_dir + "/" + app
            os.system(f"open -a '{app_dir}'")

META_OPERATION = {
    # Defining meta-operation functions
    "CLICK": click,
    "DOUBLE_CLICK": double_click,
    "RIGHT_CLICK": right_click,
    "TYPE": type_input,
    "HOVER": hover,
    "SCROLL_DOWN": scroll_down,
    "SCROLL_UP": scroll_up,
    # "SCROLL_RIGHT": ["box"], 
    # "SCROLL_LEFT": ["box"],
    "KEY_PRESS": key_press,
    "LAUNCH": launch,
    # "QUOTE_TEXT": ["box"],
    # "QUOTE_CLIPBOARD": ["output"],
    # "TEXT_FORMAT": ["input"],
    # "LLM": ["prompt"],
    "END": end,
}


def locateOnScreen(image, screenshotIm):
    print(image, screenshotIm)
    start = time.time()
    while True:
        try:
            # the locateAll() function must handle cropping to return accurate coordinates,
            # so don't pass a region here.
            retVal = pyautogui.locate(image, screenshotIm)
            try:
                screenshotIm.fp.close()
            except AttributeError:
                # Screenshots on Windows won't have an fp since they came from
                # ImageGrab, not a file. Screenshots on Linux will have fp set
                # to None since the file has been unlinked
                pass
            if retVal or time.time() - start > 0:
                return retVal
        except:
            if time.time() - start > 0:
                return None


def convert_to_meta_operation(Grounded_Operation):
    detailed_operation = {}
    if Grounded_Operation["operation"] in META_PARAMETER:
        detailed_operation["meta"] = Grounded_Operation["operation"]
        for value in META_PARAMETER[Grounded_Operation["operation"]]:
            if value in Grounded_Operation:
                if value == "box":
                    # number = (left, top, width, height)
                    numbers = Grounded_Operation["box"]
                    box = [num / 1000 for num in numbers]
                    # box = (left/1000, top/1000, width/1000, height/1000)
                    width, height = pyautogui.size()
                    # x_min, y_min, x_max, y_max = left, top, right, down)
                    x_min, y_min, x_max, y_max = [
                        int(coord * width) if i % 2 == 0 else int(coord * height)
                        for i, coord in enumerate(box)
                    ]
                    x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
                    detailed_operation[value] = (x, y)
                else:
                    detailed_operation[value] = Grounded_Operation[value][1:-1]
        print(detailed_operation)
        return detailed_operation
    else:
        raise "Wrong operation or operation not registered!"


def agent(Grounded_Operation):
    detailed_operation = convert_to_meta_operation(Grounded_Operation)
    META_OPERATION[detailed_operation["meta"]](detailed_operation)
    time.sleep(2)
    return detailed_operation["meta"]
