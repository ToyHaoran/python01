#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

def 语法错误():
    if 0:
        print("语法错误=============")
        #print(："sdf")#会标红

def 异常及异常处理():
    if 0:
        print("异常及异常处理===============")
        """
            首先，执行try子句（在关键字try和关键字except之间的语句）
            如果没有异常发生，忽略except子句，try子句执行后结束。
            如果在执行try子句的过程中发生了异常，那么try子句余下的部分将被忽略。
            如果异常的类型和 except 之后的名称相符，那么对应的except子句将被执行。最后执行 try 语句之后的代码。
            如果一个异常没有与任何的except匹配，那么这个异常将会传递给上层的try中。
        """
        try:
            print(9/0) #ZeroDivisionError
            #'2' + 2 #TypeError
            print("hello")
        except ZeroDivisionError as err:
            print("除数不能为0:{0}".format(err))
        except (RuntimeError, NameError):
            print("其他异常")
        except:
            print("无法处理，抛出异常:", sys.exc_info()[0])
            raise #可以无限往上抛
        else:
            print("没有异常")
        finally:
            print("清理行为：无论有没有异常都会执行")

def 用户自定义异常():
    if 0:
        print("用户自定义异常===================")
        class MyError(Exception):
            def __init__(self, value): # 类 Exception 默认的 __init__() 被覆盖
                self.value = value
            def __str__(self):
                return repr(self.value)
        try:
            raise MyError(2 * 2)
        except MyError as e:
            print('My exception occurred, value:', e.value)

def 调试():
    if 0:
        print("方式1：通用：把有问题的变量打印出来====")
        # print()最大的坏处是将来还得删掉它，想想程序里到处都是print()，运行结果也会包含很多垃圾信息

    if 0:
        print("方式2：断言=======")
        # 凡是用print()来辅助查看的地方，都可以用断言（assert）来替代
        def foo(s):
            n = int(s)
            assert n != 0, 'n is zero!'
            return 10 / n
        foo(0)
        # 程序中如果到处充斥着assert，和print()相比也好不到哪去

    if 0:
        print("方式3：logging=======")
        # 把print()替换为logging是第3种方式，和assert比，logging不会抛出错误，而且可以输出到文件
        import logging
        s = '0'
        n = int(s)
        logging.info('n = %d' % n)
        print(10 / n)

    if 0:
        print("方式4：适合小程序:IDE中打断点=====")

if __name__ == '__main__':
    语法错误()
    异常及异常处理()
    用户自定义异常()
    调试()
