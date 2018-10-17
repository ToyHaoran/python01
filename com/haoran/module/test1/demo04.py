#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    import com.haoran.module.test1.demo03

else:
    print("demo04模块被引入时运行")
    # 斐波那契(fibonacci)数列模块
    def fib(n):  # 定义到 n 的斐波那契数列
        a, b = 0, 1
        while b < n:
            print(b, end=' ')
            a, b = b, a + b
        print()


    # 类似_xxx和__xxx这样的函数或变量就是非公开的（private），不应该被直接引用，比如_abc，__abc等
    def _fib2(n):  # 返回到 n 的斐波那契数列
        result = []
        a, b = 0, 1
        while b < n:
            result.append(b)
            a, b = b, a + b
        return result
