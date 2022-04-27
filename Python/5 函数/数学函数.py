if __name__ == '__main__':
    import math
    print(abs(-10), math.fabs(-10), math.ceil(4.3), math.floor(4.9), math.exp(1), math.log(100, 10), pow(2, 11))
    # 10 10.0 5 4 2.718281828459045 2.0 2048
    print(round(3.14156, 3), math.sqrt(4), max(1, 2, 3), min(1, 2, 3))  # 3.142 2.0 3 1
    print(round(10.5), round(11.5))  # 10 12

    # 随机数函数：
    import random
    print(random.choice(range(100)))
    print(random.randrange(100))  # 从 0-99 选取一个随机数
    print(random.randrange(1, 100, 2))  # 从 1-99 中选取一个奇数(2是递增基数)
    list1 = [1, 2, 3, 4, 5]
    random.shuffle(list1)  # 将序列随机排序
    print(list1)  # [1, 4, 5, 3, 2]
    print(random.uniform(3, 10))  # 在3-10内随机生成实数
    print(random.randint(10, 20))  # 整数
