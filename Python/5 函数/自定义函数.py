if __name__ == '__main__':
    # 覆写x的n次幂; 默认参数必须指向不变对象
    def power(x, n=2):
        s, i = 1, n
        while i > 0:
            i -= 1
            s *= x
        return x, n, s  # 返回多个值，结果以元组形式表示
    x, n, s = power(2, 10)
    print(f"{x}的{n}次幂为{s}")

    # 参数组合顺序：必选参数aabb 默认参数cc *可变参数(元组)dd 命名关键字参数ee **关键字参数(字典)ff
    def print_strs(aa, bb, cc=0, *dd, ee, **ff):
        print(aa, bb, cc, dd, ee, ff)
    # aa bb 34 ('dd', 'dd2') ee {'ff': 'ff', 'gg': 'gg'}
    print_strs("aa", "bb", 34, "dd", "dd2", ee="ee", ff="ff", gg="gg")

    # 可接受任意参数的调用见 装饰器
