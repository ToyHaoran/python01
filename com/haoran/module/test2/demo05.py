#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    # 关于包与包之间的引用

    import sys
    print(sys.path)
    # H:\\code\\idea\\python\\com\\haoran\\module\\test2', 'H:\\code\\idea\\python'

    if 0:
        # 看到没有，路径只有本路径和项目路径，没有test1路径，所以一定要写全路径才能导入。
        import com.haoran.module.test1.demo04

    if 0:
        # 或者往路径中插入一个 (和操作列表一样)
        sys.path.insert(0, r"H:\code\idea\python\com\haoran\module\test1")
        # sys.path.append(r"H:\code\idea\python\com\haoran\module\test1")
        print(sys.path)
        import demo04


    if 0:
        print("__init__.py中的__all__属性=========")
        from com.haoran.module.test1 import * # 不推荐使用这种导包方法
        # 因为com/haoran/module/test1/__init__.py中添加了__all__属性
        # 所以只导进来了demo04包，排除了demo03
        demo04.fib(10)

    if 0:
        print("导包的路径问题==========")
        # 一个点表示同级目录 #报错No module named '__main__.demo04'
        # 当这个模块是在别的模块中被导入使用，此时的”.”就是原模块的文件名。在main函数中执行时，此时”.”变成了”__main__”。
        # 所以最好还是用绝对导入路径,安全
        # from .demo04 import fib

        # 两个点表示上级目录
        from ..test1.demo04 import fib as xxx #起别名，很有用
        xxx(100)

        #三个点表示上上级目录
        from ...module.test1.demo04 import fib as xxxx
