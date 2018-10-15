if __name__ == '__main__':
    if 0:
        print("print函数=============")
        print("hello world")
        print("hello", "你好", "我是666", sep="--", end=";") # 都是有默认选项的
        print()

    if 0:
        print("input函数=============")
        name = input("输入你的名字：")
        print("你的名字是:", name)

    if 1:
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

