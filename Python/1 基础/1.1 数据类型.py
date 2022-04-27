if __name__ == '__main__':
    # 整数 <class 'int'>
    n1, n2, n3 = 20, 0xa1b3_c3d4, 1000_000_000
    # 浮点数 <class 'float'>
    f1, f2 = 5.5, 1.2e5
    # 布尔值  <class 'bool'> 只有True、False两种值
    b1, b2 = True, False
    # 空值 None 不能理解为0
    # 变量n1 和 常量PI
    # 复数 <class 'complex'>
    ff1 = 4+3j

    print("数据类型转换===========")
    print(int('123'), int(12.34))
    print(float('12.34'))
    print(str(1.23), str(100))
    print(bool(1), bool(''))
