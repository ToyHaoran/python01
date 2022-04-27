from PIL import Image
import os
import random


def solve(src, dest):
    """
    首先读取文件夹里的所有图片，然后对这些图片进行任意位置的切割，切割尺寸是128*128，最后把切割好的图片保存在另一个文件夹里。
    :param src: 源文件夹
    :param dest: 目标文件夹
    :return: None
    """
    # 读取文件列表
    files = os.listdir(src)
    for filename in files:
        path = src + "/" + filename
        if not os.path.isdir(path):
            img = Image.open(path)
            # 进行任意位置切割128*128的矩形
            (width, high) = img.size
            left = random.randrange(width - 128)
            top = random.randrange(high - 128)
            box = (left, top, left + 128, top + 128)
            region = img.crop(box)
            # 保存文件
            region.save(dest + "/" + filename)


if __name__ == '__main__':
    src = "D:/图片/A-Z"
    dest = "D:/下载"
    solve(src, dest)
