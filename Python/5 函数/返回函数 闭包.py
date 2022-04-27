
if __name__ == '__main__':
    print("函数作为返回值================")

    # 当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中，称为闭包
    def lazy_sum(*args):
        def sum():
            ax = 0
            for n in args:
                ax = ax + n
            return ax

        return sum

    f = lazy_sum(1, 3, 5)  # 返回sum()
    print(f())  # 调用函数f时，才真正计算

    # 返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
    def count():
        fs = []
        for i in range(1, 4):
            def f():
                return i * i
            fs.append(f)
        return fs  # 返回list，包含3个函数
    f1, f2, f3 = count()
    # 原因就在于返回的函数引用了变量i，但它并非立刻执行。
    # 等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9。
    print(f1(), f2(), f3())  # 9 9 9

    # 改进：再创建一个函数，用该函数的参数绑定循环变量当前的值，无论该循环变量后续如何更改，已绑定到函数参数的值不变：
    def count():
        fs = []
        def f(n):
            def j():
                return n * n
            return j
        for i in range(1, 4):
            fs.append(f(i))  # f(i)立刻被执行，因此i的当前值被传入f()
        return fs

    f1, f2, f3 = count()
    print(f1(), f2(), f3())  # 1 4 9

    # nonlocal 使用闭包时，对外层变量赋值前，需要先使用nonlocal声明该变量不是当前函数的局部变量。
    def inc():
        x = 0
        def fn():
            nonlocal x  # 解释器把fn()的x看作外层函数的局部变量，它已经被初始化了，可以正确计算x+1。
            x = x + 1  # x作为局部变量并没有初始化，直接计算x+1是不行的，必须加nonlocal
            return x
        return fn
    f = inc()
    print(f())  # 1
    print(f())  # 2
