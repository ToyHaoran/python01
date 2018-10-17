#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""a test module"""

__author__ = 'lihaoran'

# 第1行和第2行是标准注释，第1行注释可以让这个hello.py文件直接在Unix/Linux/Mac上运行，第2行注释表示.py文件本身使用标准UTF-8编码；
# 第4行是一个字符串，表示模块的文档注释，任何模块代码的第一个字符串都被视为模块的文档注释；
# 第6行使用__author__变量把作者写进去，这样当你公开源代码后别人就可以瞻仰你的大名；
# 以上就是Python模块的标准文件模板，当然也可以全部删掉不写，但是，按标准办事肯定没错

if __name__=="__main__":
    # 如果在其他地方导入该模块时，if判断将失败
    print("demo03==========")
    print('程序自身在运行')
    if 0:
        print("导包相关操作==================")
        import sys # import sys 引入 python 标准库中的 sys.py 模块；这是引入某一模块的方法
        for i in sys.argv: # sys.argv 是一个包含命令行参数的列表
            print(i)
        sys.path.insert(0,r'C:\code\project\com')
        print ('python 路径为',sys.path) # sys.path 包含了一个 Python 解释器自动查找所需模块的路径的列表，注意是一个list
        # C:\\code\\project\\com', 'H:\\code\\idea\\python\\com\\haoran\\module\\test1', 'H:\\code\\idea\\python'


    if 0:
        #从顶级包开始导入，绝对路径
        print("导入自定义的包===============")
        import com.haoran.module.test1.demo04 as d4 # 可以重命名
        d4.fib(100)
        # print(d4.fib2(100)) # fib2是私有函数，无法引用，只能在demo4模块内用。

    if 0:
        print("from … import 语句==============")
        from com.haoran.module.test1.demo04 import fib
        fib(100)

    if 0:
        import com.haoran.module.test1.demo04 as demo04
        import sys
        print("dir()函数==============")
        # 内置的函数 dir() 可以找到模块内定义的所有名称。以一个字符串列表的形式返回:
        print(dir(demo04))
        print(dir(sys))

else:
    print("demo03模块被引入时运行")
