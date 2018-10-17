#! /usr/bin/env python
# -*- coding: utf-8 -*-

if __name__=="__main__":
    print("demo03==========")
    print('程序自身在运行')
    if 1:
        print("导包相关操作==================")
        import sys # import sys 引入 python 标准库中的 sys.py 模块；这是引入某一模块的方法
        for i in sys.argv: # sys.argv 是一个包含命令行参数的列表
            print(i)
        sys.path.insert(0,r'C:\code\project03python\com')
        print ('python 路径为',sys.path) # sys.path 包含了一个 Python 解释器自动查找所需模块的路径的列表

    if 0:
        #语句较长
        print("导入自定义的包===============")
        import src.com.test1.demo04
        src.com.test1.demo04.fib(100)
        print(src.com.test1.demo04.fib2(100))

    if 0:
        #推荐  语句较短
        print("from … import 语句==============")

        #从顶级包开始导入，绝对路径
        from com.haoran.module.test1.demo04 import  fib
        fib(100)

        #一个点表示同级目录 #报错No module named '__main__.demo04'
        #当这个模块是在别的模块中被导入使用，此时的”.”就是原模块的文件名。在main函数中执行时，此时”.”变成了”__main__”。
        #所以最好还是用绝对导入路径,安全
        #from .demo04 import fib

        #两个点表示上级目录
        #from ..test1.demo04 import fib as xxx #起别名，很有用
        #xxx(100)

        #三个点表示上上级目录
    if 0:
        import demo04, sys
        print("dir()函数==============")
        # 内置的函数 dir() 可以找到模块内定义的所有名称。以一个字符串列表的形式返回:
        print(dir(demo04))
        print(dir(sys))

else:
    print("demo03模块被引入时运行")
