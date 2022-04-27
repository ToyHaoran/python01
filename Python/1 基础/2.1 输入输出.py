if __name__ == '__main1__':
    name = input("输入你的名字：")  # input()返回的数据类型是str
    print("你好", name)  # 逗号表示一个空格
    print("你好", name, sep="--", end=";\n")  # 有默认选项的
    print(f"你好:{name}")


if __name__ == '__main__':
    # 输入两个整数或字符串
    x, y = eval(input("请输入2个整数，中间用逗号分开："))
    n, m = input("请输入2个字符串，中间用逗号分开：").split(',')
    print(type(x), type(y), type(m), type(n))
