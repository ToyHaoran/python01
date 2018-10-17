#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python对函数式编程提供部分支持。由于Python允许使用变量，因此，Python不是纯函数式编程语言

import time

def 高阶函数():
    if 0:
        print("函数作为参数==========")
        f = abs
        def add(x, y, f):
            return f(x) + f(y)

        print(add(-5, 6, abs)) # 11

def map函数():
    if 0:
        print("map================")
        # map()函数接收两个参数，一个是函数，一个是Iterable，map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回。
        def f(x):
            return str(x) + "的平方是：" + str(x * x) + " "
        r = map(f, list(range(1,10)))
        print(r.__next__()) # 1的平方是：1
        print(list(r))

def reduce函数():
    if 0:
        print("reduce===============")
        from functools import reduce
        # 把序列[1, 3, 5, 7, 9]变换成整数13579
        def fn(x, y):
            return x * 10 + y
        res = reduce(fn, list(range(1, 10, 2)))
        print(res) # 13579

def map_reduce练习():
    if 0:
        print("练习=============")
        # 1、利用map()函数，把用户输入的不规范的英文名字，变为首字母大写，其他小写的规范名字。
        # 输入：['adam', 'LISA', 'barT']，输出：['Adam', 'Lisa', 'Bart']：
        print(list(map(lambda str1: str1.title(), ['adam', 'LISA', 'barT'])))

        # 2、请编写一个prod()函数，可以接受一个list并利用reduce()求积：
        from functools import reduce
        print(reduce(lambda x, y: x * y, [3, 5, 7, 9]))

        # 3、利用map和reduce编写一个str2float函数，把字符串'123.456'转换成浮点数123.456
        def str2float(s):
            DIGITS = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': ''}
            n = 0 # 整数部分的位数
            for x in s:
                n = n + 1
                if x == '.':
                    s1 = s[:n - 1] # 整数部分
                    s2 = s[n:] # 小数部分
                    break
            def fn(x, y):
                return x * 10 + y
            def char2num(s):
                return DIGITS[s]
            return reduce(fn, map(char2num, s1 + s2)) / (10 ** (len(s) - n))
        res = str2float("123.456")
        print(abs(res - 123.456) < 0.000001)

def filter函数():
    if 0:
        # 和map()类似，filter()也接收一个函数和一个序列。
        # 和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
        print("用filter求素数============")
        # 计算素数的一个方法是埃氏筛法，它的算法理解起来非常简单：
        # 首先，列出从2开始的所有自然数，构造一个序列：
        # 取序列的第一个数2，它一定是素数，然后用2把序列的2的倍数筛掉：
        # 取新序列的第一个数3，它一定是素数，然后用3把序列的3的倍数筛掉：
        # 取新序列的第一个数5，然后用5把序列的5的倍数筛掉：
        # 不断筛下去，就可以得到所有的素数。

        # 用Python来实现这个算法，可以先构造一个从3开始的奇数序列：(偶数不可能，有2)
        def _odd_iter():
            n = 1
            while True:
                n = n + 2
                yield n
        # 然后定义一个筛选函数：
        def _not_divisible(n):
            return lambda x: x % n > 0
        # 最后，定义一个生成器，不断返回下一个素数
        def primes():
            yield 2
            it = _odd_iter()  # 初始序列
            while True:
                n = next(it)  # 返回序列的第一个数 （第一次返回3）
                yield n # 第一次返回3暂停
                it = filter(_not_divisible(n), it)  # 构造新序列 (第一次把it中3的倍数都过滤掉)

        # 打印1000以内的素数:
        for n in primes():
            if n < 1000:
                print(n, end=" ")
            else:
                break

def 排序算法():
    if 0:
        # key指定的函数将作用于list的每一个元素上，并根据key函数返回的结果进行排序
        print(sorted([36, 5, -12, 9, -21]))
        print(sorted([36, 5, -12, 9, -21], key=abs))

        # 默认情况下，对字符串排序，是按照ASCII的大小比较的
        # 实现忽略大小写的排序
        print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower))
        # 降序排列
        print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True))

        students = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
        from operator import itemgetter
        print(sorted(students, key=itemgetter(1)))
        print(sorted(students, key=lambda t: t[1])) # Student未改变
        # 或
        students.sort(key=lambda t: t[1]) # 返回None，Student已经改变
        print(students)

def 函数作为返回值():
    if 0:
        def lazy_sum(*args):
            def sum():
                ax = 0
                for n in args:
                    ax = ax + n
                return ax
            return sum
        f = lazy_sum(1, 3, 5) # 返回sum()
        f2 = lazy_sum(2, 4, 6) # 返回sum()
        print(f())
        print(f2())
        # 在这个例子中，我们在函数lazy_sum中又定义了函数sum，并且，内部函数sum可以引用外部函数lazy_sum的参数和局部变量，
        # 当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，这种称为“闭包（Closure）”的程序结构拥有极大的威力

def 闭包():
    if 0:
        print("使用循环创建同时创建多个函数========")
        print("错误案例======")
        def count():
            fs = []
            for i in range(1, 4):
                def f():
                    return i*i
                fs.append(f)
            return fs
        # 每次循环，都创建了一个新的函数，然后，把创建的3个函数都返回了。
        # 你可能认为调用f1()，f2()和f3()结果应该是1，4，9，但实际结果是9,9,9
        f1, f2, f3 = count()
        print(f1())
        print(f2())
        print(f3())
        # 全部都是9！原因就在于返回的函数引用了变量i，但它并非立刻执行。等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9。
        # 返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
        # 解决方式：
        print("正确案例========")
        def count():
            def f(j):
                def g():
                    return j*j
                return g
            fs = []
            for i in range(1, 4):
                fs.append(f(i)) # f(i)立刻被执行，因此i的当前值被传入f()
            return fs
        f1, f2, f3 = count()
        print(f1())
        print(f2())
        print(f3())

        # 缺点是代码较长，可利用lambda函数缩短代码。

def Lambda表达式():
    if 0:
        print("Lambda表达式（匿名函数）=============")
        # 匿名函数有个限制，就是只能有一个表达式，不用写return，返回值就是该表达式的结果。
        def make_sum():
            return lambda arg1, arg2: arg1 + arg2
        sum = make_sum() # sum是一个函数
        print("相加后的值为 : ", sum(10, 20))

def 装饰器():
    if 1:
        def now():
            print("现在")
        # 现在，假设我们要增强now()函数的功能，
        # 比如，在函数调用前后自动打印日志，但又不希望修改now()函数的定义，
        # 这种在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）。

        if 0:
            # 我们要定义一个能打印日志的decorator
            # 因为它是一个decorator，所以接受一个函数作为参数，并返回一个函数
            # 我们要借助Python的@语法，把decorator置于函数的定义处：now()上面
            # 必须将log函数放在now函数上面。
            def log(func):
                def wrapper(*args, **kw):
                    print('call %s():' % func.__name__)
                    return func(*args, **kw)
                return wrapper

            @log
            def now():
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            now() # now = log(now)

        if 1:
            # 如果decorator本身需要传入参数，那就需要编写一个返回decorator的高阶函数，写出来会更复杂。比如，要自定义log的文本
            def log(text):
                def decorator(func):
                    def wrapper(*args, **kw):
                        print('%s %s():' % (text, func.__name__))
                        return func(*args, **kw)
                    return wrapper
                return decorator

            @log('execute')
            def now():
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            now() #  now = log('execute')(now)
            # 首先执行log('execute')，返回的是decorator函数，再调用返回的函数，参数是now函数，返回值最终是wrapper函数


if __name__ == '__main__':
    高阶函数()
    map函数()
    reduce函数()
    map_reduce练习()
    filter函数()
    排序算法()
    函数作为返回值()
    Lambda表达式()
    闭包()
    装饰器()

