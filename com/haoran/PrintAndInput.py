
def 输出函数():
    if 0:
        print("print函数=============")
        print("hello world")
        print("hello", "你好", "我是666", sep="--", end=";") # 都是有默认选项的
        print()

def 输入函数():
    if 0:
        print("input函数=============")
        name = input("输入你的名字：")
        print("你的名字是:", name)

def 行与缩进():
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

    if 0:
        """
            Python两种输出值的方式: 表达式语句和 print() 函数。
            第三种方式是使用文件对象的 write() 方法，标准输出文件可以用 sys.stdout 引用。
            如果你希望输出的形式更加多样，可以使用 str.format() 函数来格式化输出值。
            如果你希望将输出的值转成字符串，可以使用 repr() 或 str() 函数来实现。
            str()： 函数返回一个用户易读的表达形式。
            repr()： 产生一个解释器易读的表达形式。
        """
        print("输出一个平方与立方的表==========================")
        print("rjust()方法, 它可以将字符串靠右, 并在左边填充空格,类似的ljust() 和 center()")
        for x in range(1, 11):
            print(repr(x).rjust(2), repr(x*x).rjust(3), repr(x*x*x).rjust(4))
        for x in range(1, 11):
            print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))

        print("zfill(), 它会在数字的左边填充0============")
        print('-3.14'.zfill(7)) #-003.14

    if 0:
        print("读取键盘输入=================")
        print ("input is: ", input("input："))

    if 0:
        print("文件读写=================")
        #绝对路径

        print("写入文件============")
        f = open("C:/code/project03python/resource/foo.txt", "w")
        # 默认保存为GBK编码，不知道为什么？
        num = f.write("Python 是一个非常好的语言。\n是的，的确非常好!!\n好NMLD") #写入字符串
        # print(num)#写入字符数
        # f.write(str(("你好",14))) #写入其他数据类型，需要转换为字符串
        # print(f.tell())#返回文件对象当前所处的位置, 它是从文件开头开始算起的字节数
        #print(f.seek(3))  #seek(x,0) ：从起始位置即文件首行首字符开始移动 x 个字符 default，其他不知道为什么不行。


        # print("读取文件===========")
        # f = open("C:/code/project03python/resource/foo.txt", "r")
        # print(f.read())#读取一个文件的内容
        # print(f.readline())#读取单独的一行
        # print(f.readlines())#读取所有行
        # for line in f: print(line, end="")#迭代文件对象，读取每一行

        f.close()

    if 1:
        print("pickle 模块=============")#跳过


if __name__ == '__main__':
    输出函数()
    输入函数()
    行与缩进()