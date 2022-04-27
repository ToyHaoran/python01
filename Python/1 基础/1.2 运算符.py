if __name__ == '__main__':
    # 算数运算符
    print("a" * 10)  # aaaaaaaaaa
    print(2 ** 11)  # 幂运算符  2048
    print(5 / 2)  # 除法 2.5
    print(5 // 2)  # 取整 2
    print(5.0 // 2)  # 2.0 不同类型运算会将整数转换为浮点数
    print(5 // 2.0)  # 2.0
    print(5 % 2)  # 求余 1
    # 比较运算符：没什么不同
    # 赋值运算符：就是把算数运算符后面加个=

    # 位运算符：
    a = 60  # 0011 1100
    b = 13  # 0000 1101
    print(a & b, a | b, a ^ b, ~a, a << 2, a >> 2)  # 12 61 49 -61 240 15

    # 逻辑运算符
    a = 0
    b = 20
    c = 10
    print(a and b)  # 0 如果 x 为 False，x and y 返回 False，否则它返回 y 的计算值。
    print(a or c)  # 10 如果 x 是 True，它返回 x 的值，否则它返回 y 的计算值。
    print(not a)  # True 如果 x 为 True，返回 False 。如果 x 为 False，它返回 True。

    # 成员运算符
    a, b = 1, 20
    list1 = [1, 2, 3, 4, 5]
    print(a in list1)
    print(b not in list1)

    # 身份运算符
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(a is b)  # 判断是否为同一个对象 False
    print(id(a) == id(b))  # 同上
    print(a == b)  # 判断值是否相等 True

