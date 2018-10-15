# ————————————————————————————基本数据类型
if 1:
    # ————————————Number（数字）
    if 1:
        a = b = c = 1
        print(a + b + c)  # 3
        a, b, c, d = 20, 5.5, True, 4 + 3j  # 4+3j是一个复数
        print(type(a), type(b), type(c), type(d))  # <class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
        print(isinstance(a, int))  # True
        # type()不会认为子类是一种父类类型。
        # isinstance()会认为子类是一种父类类型。

        print("——————算数运算符")
        print("a" * 10)
        print(2 ** 11)  # 幂运算符  2048
        print(5 / 2)  # 2.5
        print(5 // 2)  # 取整 2
        print(5.0 // 2)  # 2.0 不同类型运算会将整数转换为浮点数
        print(5 // 2.0)  # 2.0
        print(5 % 2)  # 1

        print("————————数学函数")
        import math

        print(abs(-10), math.fabs(-10), math.ceil(4.3), math.floor(4.9), math.exp(1), math.log(100, 10), pow(2, 11))
        print(round(3.14156, 3), math.sqrt(4), max(1, 2, 3), min(1, 2, 3))
        print(math.modf(3.424))
        # 注意一个坑
        print(round(10.5), round(11.5))  # 10 12

        print("————————随机数函数")
        import random

        print(random.choice(range(100)))
        # 从 1-99 中选取一个奇数(2是递增基数)
        print(random.randrange(1, 100, 2))
        # 从 0-99 选取一个随机数
        print(random.randrange(100))
        list = [1, 2, 3, 4, 5]
        random.shuffle(list)  # 将序列随机排序
        print(list)
        print(random.uniform(3, 10))  # 在3-10内随机生成实数
        print(random.randint(10, 20))  # 整数

        print("————————三角函数，跳过")
