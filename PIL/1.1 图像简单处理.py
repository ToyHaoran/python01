import os
from PIL import Image  # 调用库

"""
  Python Imaging Library (PIL)：是python下的图像处理模块,支持多种格式,并提供强大的图形与图像处理功能。
  Python的快速开发能力以及面向对象等等诸多特点使得它非常适合用来进行原型开发，以及对于简单的图像处理或者大批量的简单图像处理任务。
  Pillow 是友好的 PIL 复刻；
"""

if __name__ == '__main__':
    # 读取 压缩 另存为
    path = "D:/图片/A-Z/A.jpg"
    files = os.listdir("D:/图片/A-Z")  # 该文件下所有文件名称
    im = Image.open(path)  # 加载图片，\也能用，但是与转义字符冲突；
    # print(im.format, im.size, im.mode)  # 图片属性 JPEG (1366, 768) RGB
    # im.show()  # 用默认程序显示图像
    f, e = os.path.splitext(path)  # ('D:/图片/A-Z/A', '.jpg')
    # im.thumbnail((128, 128))  # 压缩图片(像素)
    # im.save(f+".png")  # 另存为png图片
    # Image.new("RGB", (2000, 768))  # 创建白板

    # 剪切 处理 粘贴矩形 (滚动图像 合并图像)
    box = (500, 100, 800, 400)  # 坐标(左、上、右、下) 左上角是(0,0)
    region = im.crop(box)  # 从图像中复制矩形
    region = region.transpose(Image.ROTATE_180)  # 处理矩形，旋转180度
    region = region.rotate(45)  # 逆时针旋转45度
    region = region.resize((400, 400))
    # im.paste(region, box)  # 粘贴子矩形
    im.show()
