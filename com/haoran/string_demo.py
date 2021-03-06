#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 第一行注释是为了告诉Linux/OS X系统，这是一个Python可执行程序，Windows系统会忽略这个注释；
# 第二行注释是为了告诉Python解释器，按照UTF-8编码读取源代码，否则，你在源代码中写的中文输出可能会有乱码。
# 大部分.py文件不必以#!作为文件的开始.
# 根据 PEP-394 , 程序的main文件应该以 #!/usr/bin/python2或者 #!/usr/bin/python3开始.

字符串相关操作 = 0
if 0:
    print("字符串相关操作==============")
    word = "I am "
    name = 'lihaoran'
    print(r'"""实现多行字符串=========')
    # 三引号，常用语引用HTML代码，或SQL代码
    helloword = ("""\
        hello
        world""")
    print(helloword)

    # 如果代码太长写成一行不便于阅读 可以使用\或()折行
    str1 = word + \
          name
    print(str1)

    print("理解字符串的内存========")
    a = 'ABC'
    b = a
    a = 'XYZ'
    print(b) # 'ABC'

    print("转义字符============")
    print("aaa\\bbb\'ccc\"ddd\teee\nfff")

    print("不使用转义字符(r表示原始字符串，不发生转义)=========")
    print(r"xxx\nlll")

    print("字符串拼接==============")
    a = "hello"
    b = "world"
    c = 21
    print(a + b)  # helloword  注意+号连接大量字符串是非常低效的。
    print(a, c)  # hello 21 默认以空格分隔
    # print(a + c)# 错误，不能将int转为String
    print(a + str(c))  # 如果报错str不可用，说明你把内置的函数给占用了。


字符串编码 = 0
if 0:
    print("字符串编码===============")
    # 参考：https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431664106267f12e9bef7ee14cf6a8776a479bdec9b9000
    # 在Python3中，所有的字符串都是Unicode字符串。
    print(ord("A")) # 65 ord()函数获取字符的整数表示
    print(chr(65)) # A chr()函数把编码转换为对应的字符
    print('\u4e2d\u6587') # 中文

    # 如果要在网络上传输，或者保存到磁盘上，就需要把str变为以字节为单位的bytes
    x = b'abc' # 每个字符都只占用一个字节
    print('ABC'.encode('ascii'))
    print(b'ABC'.decode('ascii'))

    print('中文'.encode('utf-8')) # 长度为6
    print(b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8')) # 长度为2


字符串访问 = 0
if 0:
    print("字符串访问，截取==========")
    #索引值以 0 为开始值，-1 为从末尾的开始位置。
    #从后面索引：-6 -5 -4 -3 -2 -1
    #从前面索引： 0  1  2  3  4  5
    #            a  b  c  d  e  f
    #从后面截取：  1  2  3  4  5
    #从前面截取： -5 -4 -3 -2 -1
    str = 'Runoob'
    print(str)  # 输出字符串
    print(str[-1])  # 输出倒数第一个字符 b
    print(str[0])  # 输出字符串第一个字符 R
    print("注意：切片总是左毕右开原则")
    print(str[2:5])  # noo
    print(str[:3] + " " + str[3:]) #Run oob
    print(str * 2)  # 输出字符串两次 RunoobRunoob
    # Python 没有单独的字符类型，一个字符就是长度为1的字符串。
    print(str[0] == str[0:1]) # True
    word = 'Python'
    # 检查是否是子字符串
    print('P' in word)


字符串格式化 = 0
# 目前有三种方式：
if 0:
    print("方式1：字符串前面加个f==========")
    year = 2018
    mon = 10
    # 能运行，但是语法上报错,推测是代码检查的工具还没有更新（3.6更新的）
    # print(f"3+3的结果是{3+3}")
    # print(f"今天是{year}年{mon}月")


    # table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
    # for name, phone in table.items():
    #     print(f'{name:10} ==> {phone:10d}')

    # 使用的{phone:10d}等格式和format是相同的。
    # 参考：https://docs.python.org/3.7/library/string.html#formatspec

if 0:
    print("方式2：str.format()方法==========")

    # 通过位置设置参数
    print("{1} {0} {1}".format("hello", "world"))
    # 通过字典设置参数
    site = {"name": "菜鸟教程", "url": "www.runoob.com"}
    print("网站名：{name}, 地址：{url}".format(**site)) # 解压
    print("网站名：{0[name]}, 地址：{0[url]}".format(site))
    # 通过列表索引设置参数
    my_list = ['菜鸟教程', 'www.runoob.com', 4444]
    print("网站名：{0}, 地址：{1}, 价钱：{2:d}".format(*my_list)) # 解压
    print("网站名：{0[0]}, 地址：{0[1]}, 价钱：{0[2]:b}".format(my_list))  # "0" 是必须的

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
        print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

if 0:
    print("方式3：%方法（已经过时,但是能用）======")
    # 参考：https://docs.python.org/3.7/library/stdtypes.html#old-string-formatting
    print("我叫 %s 今年 %d 岁!" % ('小明', 10))#我叫 小明 今年 10 岁!

    # 对数字进行格式化
    # 百分号的用法%[(name)][flags][width].[precision]typecode
    print("%6.3f" % 2.3)  # 宽度为6，小数3位，右对齐，浮点型，前面有一空格 2.300
    print("%.2f" % 2.232)  # 保留两位小数
    print("%+5x" % -10)  # 右对齐，宽度5，前面3空格
    import math
    print("pi的值是%s" % math.pi)
    print("%10.*f" % (4, 1.2))  # 1.2000


字符串内建函数 = 0
if 0:
    print("字符串内建函数====================")
    # 参考：
    # https://docs.python.org/3.7/library/stdtypes.html#string-methods
    str1 = "hello"
    str2 = "123"
    print(str1.capitalize())  # 首字母大写
    print(str1.center(9, "-"))
    print(str1.count("ll", 0, 5))  # 字母出现的次数
    print(str1.startswith("he", 0, 10))  # 字符串是不是以he开头
    print(str1.endswith("lo", 0, 10))  # 字符串是不是以lo结尾
    print(str1.find("e"))  # 1
    print(str1.rfind("e"))  # 1
    print(str1.index("e"))
    print(str1.rindex("e"))
    print((str1 + str2).isalnum())  # 如果字符串中的所有字符都是字母数字且至少有一个字符，则返回true，否则返回false
    print(str2.isdigit())
    print(str1.isalpha())
    print(str1.islower())
    print(str1.isupper())  # 字符串是不是大写
    print(str1.istitle())
    print(len(str1))
    str3 = "-"
    print(str3.join(("aaa", "bbb", "ccc")))
    print(str1.lower())
    print(str1.title())  # 单词首字母大写,其他变为小写
    print(str1.upper())  # 字符串变大写
    print(str1.swapcase())
    print(str1.ljust(10, "-"))
    print(str1.rjust(10, "-"))
    print("12".zfill(5))  # str.zfill()用零填充左侧的数字字符串
    print(str1.lstrip("he"))
    print(str1.rstrip("lo"))
    print(str1.strip("o"))
    print(max(str1))
    print(min(str1))
    print(str1.replace("he", "HE", 1))
    print(str1.split("e"))


def 文档字符串(aa):
    """
    测试文档注释

    一个文档字符串应该这样组织:
    首先是一行以句号, 问号或惊叹号结尾的概述(或者该文档字符串单纯只有一行).
    接着是一个空行.
    接着是文档字符串剩下的部分, 它应该与文档字符串的第一行的第一个引号对齐.

    :param aa: 传入的参数
    :return:  不返回任何东西
    """
    if 0:
        print("文档字符串=====")
        print(aa)


if __name__ == '__main__':
    文档字符串("sss")
