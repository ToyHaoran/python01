
行与缩进 = 0
if 0:
    print("行与缩进=========")
    """
        行与缩进
        python最具特色的就是使用缩进来表示代码块，不需要使用大括号 {} 。
        像if、while、def和class这样的复合语句，首行以关键字开始，以冒号( : )结束，该行之后的一行或多行代码构成代码组。
    """
    if True:
        print("true")
    else:
        print("false")
    print("另一个代码块")

输出函数 = 0
if 0:
    print("print函数=============")
    print("hello world")
    print("hello", "你好", "我是666", sep="--", end=";")  # 都是有默认选项的
    print()


输入函数 = 0
if 0:
    print("input函数=============")
    name = input("输入你的名字：")
    print("你的名字是:", name)

    # input()返回的数据类型是str
    s = input('birth: ')
    birth = int(s) # 必须要转化一下
    if birth < 2000:
        print('00前')
    else:
        print('00后')
if 0:
    user_input = input('输入一个列表，用逗号隔开:\n').strip()
    list1 = [int(item) for item in user_input.split(',')]
    print(list1)


文件读写 = 0
if 0:
    print("文件读写=================")
    # 绝对路径

    print("写入文件============")
    f = open("C:/code/project03python/resource/foo.txt", "w")
    # 默认保存为GBK编码，不知道为什么？
    num = f.write("Python 是一个非常好的语言。\n是的，的确非常好!!\n好NMLD")  # 写入字符串
    # print(num)#写入字符数
    # f.write(str(("你好",14))) #写入其他数据类型，需要转换为字符串
    # print(f.tell())#返回文件对象当前所处的位置, 它是从文件开头开始算起的字节数
    # print(f.seek(3))  #seek(x,0) ：从起始位置即文件首行首字符开始移动 x 个字符 default，其他不知道为什么不行。

    # print("读取文件===========")
    # f = open("C:/code/project03python/resource/foo.txt", "r")
    # print(f.read())#读取一个文件的内容
    # print(f.readline())#读取单独的一行
    # print(f.readlines())#读取所有行
    # for line in f: print(line, end="")#迭代文件对象，读取每一行
    f.close()
