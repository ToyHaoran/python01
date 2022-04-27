if __name__ == '__main__':
    #  方式1：字符串前面加个f  推荐
    print(f"3+3的结果是{3 + 3}")
    year, mon = 2018, 10
    print(f"今天是{year}年{mon}月")  # 今天是2018年10月
    table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
    for name, phone in table.items():
        print(f'{name:10} ==> {phone:10d}')

    #  方式2：str.format()
    print("{1} {0} {1}".format("hello", "world"))  # 通过位置设置参数
    site = {"name": "菜鸟教程", "url": "www.runoob.com"}  # 通过字典设置参数
    print("网站名：{name}, 地址：{url}".format(**site))  # 解压
    my_list = ['菜鸟教程', 'www.runoob.com', 4444]  # 通过列表索引设置参数
    print("网站名：{0}, 地址：{1}, 价钱：{2:d}".format(*my_list))  # 解压

    # 对数字进行格式化
    print("{:.2f}".format(3.1415926))  # 保留两位小数
    print("{:+.2f}".format(-3.1415926))  # 带符号保留两位小数
    print("{:.0f}".format(3.14))  # 不保留小数
    print("{:0>2d}".format(5))  # 数字补零，填充左边，宽度为2
    print("{:x<4d}".format(10))  # 数字补x，填充右边，宽度为4
    print("{:,}".format(1000000))  # 逗号分隔
    print("{:.2%}".format(0.25))  # 百分比格式，两位小数
    print("{:.2e}".format(1000000))  # 指数记法 1.00e+06
    print("{:10d}".format(13))  # 右对齐，保持10位
    print("{:>10d}".format(13))  # 右对齐，保持10位
    print("{:<10d}".format(13))  # 左对齐
    print("{:^10d}".format(13))  # 居中对齐
    print('{:b}'.format(11))  # 二进制
    print('{:d}'.format(11))  # 十进制
    print('{:o}'.format(11))  # 八进制
    print('{:x}'.format(11))  # 十六进制 b
    print('{:#x}'.format(11))  # 0xb
    print('{:#X}'.format(11))  # 0XB
    # 左边填充空格
    for x in range(1, 11):
        print('{0:2d} {1:3d} {2:4d}'.format(x, x * x, x * x * x))
    print("方式3：%方法(已经过时,但是能用)======")
    print("我叫 %s 今年 %d 岁!" % ('小明', 10))  # 我叫 小明 今年 10 岁!

    # 百分号的用法%[(name)][flags][width].[precision]typecode
    print("%6.3f" % 2.3)  # 宽度为6，小数3位，右对齐，浮点型，前面有一空格 2.300
    print("%.2f" % 2.232)  # 保留两位小数
    print("%+5x" % -10)  # 右对齐，宽度5，前面3空格
