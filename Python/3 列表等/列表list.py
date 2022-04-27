"""
列表就是Python中的数组，但可以动态调整大小，可以包含不同类型的元素；
"""
if __name__ == '__main__':
    print("list访问 切片 修改==============")
    # 从后面索引：-6 -5 -4 -3 -2 -1
    # 从前面索引： 0  1  2  3  4  5
    #             a  b  c  d  e  f
    list1 = list(range(20))
    print(list1, list1[:])  # 输出完整列表 等价
    print(list1[1:3])  # 从下标1开始输出到下标2 [左毕右开)
    print(list1[-3:-1])  # 输出后三个数 -1最后一个数
    print(list1[:10:2])  # 每两个取一个
    list1[2:5] = []  # 将对应的元素值设置为 []

    print("列表连接和重复===============")
    list2 = list(range(5))
    list3 = list1 + list2 * 2  # 要新建一个列表，并且要复制对象，计算量大。建议使用extend追加元素
    print(list3)

    print("列表遍历=========================")
    list1 = list(range(20))
    for x in list1:
        print(x, end="  ")  # 不换行输出
    for idx, item in enumerate(list1):  # 带索引 类似tuple
        print(idx, item)

    print("同时遍历两个或更多的列表序列，使用zip()=========")
    index1 = list(range(1, 4))
    for index, list2 in zip(index1, list1):
        print(f"{index}：{list2}")

    print("列表生成式：将一种数据转为另一种数据=========================")
    print([x ** 2 for x in range(20) if x % 2 == 0])  # 生成偶数x的平方
    print([x if x % 2 == 0 else 0 for x in range(20)])  # 将奇数变为0
    print([m + n for m in "ABC" for n in "XYZ"])  # 两层循环生成全排列

    print(list(map(lambda x: x ** 2, range(10))))
    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    print([x ** 2 for x in range(10)])  # 和上面等价

    print("list基本方法(可以当堆栈使用)==============")
    list2 = list(range(10))
    list2.append([3, 4])  # 看为1个元素list
    list2.extend([3, 4])  # 看做2个元素
    print(list2.index(3, 4))  # 从第4个位置开始查找3的下标
    list2.insert(3, 666)
    list2.pop(1)  # 移除下标为1的元素  默认移除最后一个元素
    list2.remove(666)  # 移除第一个值为666的元素
    list2.reverse()  # 反转
