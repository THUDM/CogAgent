import os
import cv2
import time
import shutil
import numpy as np

from skimage.metrics import structural_similarity


from controller import get_screenshot


def wait_screen_loading(adb_path, time_sleep=3, theshold=0.80):
    """
    每3秒截图，对比前一张截图，如果大致相同则认为加载完成
    Params:
        adb_path: adb路径
        time_sleep: 每次截图间隔时间
        theshold: ssim阈值，如果大于该值则认为加载完成
    """
    # 确保time_sleep大于0 且为数字
    assert (
        isinstance(time_sleep, (int, float)) and time_sleep > 0
    ), "time_sleep must be a positive number"

    # 确保theshold大于0 且小于1
    assert (
        isinstance(theshold, (int, float)) and 0 < theshold < 1
    ), "theshold must be a number between 0 and 1"

    iter = 0
    last_screenshot_file = "./screenshot/screenshot_last.jpg"
    now_screenshot_file = "./screenshot/screenshot_now.jpg"
    get_screenshot(adb_path, print_flag=False, round_num="last")  # 获取屏幕截图
    while iter < 10:
        time.sleep(time_sleep)
        get_screenshot(adb_path, print_flag=False, round_num="now")  # 获取屏幕截图

        # 对比 last_screenshot_file 和 now_screenshot_file
        # 如果相同，则认为加载完成，退出循环
        if ssim(last_screenshot_file, now_screenshot_file) > theshold:
            print("屏幕内容加载完成！")
            break

        shutil.move(now_screenshot_file, last_screenshot_file)
        iter += 1


def funnnn(adb_path):
    # 每3秒截图，对比前一张截图，如果大致相同则认为加载完成
    screenshot_file = "./screenshot/screenshot.jpg"
    get_screenshot(adb_path, print_flag=False)  # 获取屏幕截图
    last_screenshot_file = "./screenshot/screenshot_last.jpg"
    now_screenshot_file = "./screenshot/screenshot_now.jpg"
    if os.path.exists(last_screenshot_file):
        os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)
    while True:
        time.sleep(3)
        get_screenshot(adb_path, print_flag=False)  # 获取屏幕截图
        if os.path.exists(now_screenshot_file):
            os.remove(now_screenshot_file)
        os.rename(screenshot_file, now_screenshot_file)
        # 对比 last_screenshot_file 和 now_screenshot_file
        # 如果相同，则认为加载完成，退出循环
        # 调整阈值
        if ssim(last_screenshot_file, now_screenshot_file) > 0.80:
            print("屏幕内容加载完成！")
            break
        else:
            if os.path.exists(last_screenshot_file):
                os.remove(last_screenshot_file)
            os.rename(now_screenshot_file, last_screenshot_file)


def ssim(file1, file2):

    # 加载两张图片
    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)

    # 图片转换为灰度
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算结构相似性
    (score, diff) = structural_similarity(gray1, gray2, full=True)
    return score


if __name__ == "__main__":
    path1 = "./screenshot/screenshot_1.jpg"
    path2 = "./screenshot/screenshot_1.jpg"

    print(f"{ssim(path1, path2)=}")
    wait_screen_loading("adb -s hyiz5hjvz975mrdq")
