def 元组():
    if 1:
        # 元组（tuple）与列表类似，不同之处在于元组的元素不能修改。元组写在小括号 () 里，元素之间用逗号隔开。
        # 元组中的元素类型也可以不相同：
        print("元组的获取=========")
        tuple01 = ('abcd', 786, 2.23, 'runoob', 70.2)
        tinytuple = (123, 'runoob')
        print(tuple01)  # 输出完整元组
        print(tuple01[0])  # 输出元组的第一个元素
        print(tuple01[1:3])  # 输出从第二个元素开始到第三个元素
        print(tuple01[2:])  # 输出从第三个元素开始的所有元素
        print(tinytuple * 2)  # 输出两次元组
        print(tuple01 + tinytuple)  # 连接元组
        tup1 = ()  # 空元组
        tup2 = (20,)  # 一个元素的元组，需要在元素后添加逗号。千万不要这样定义t = (1)这是一个加了小括号的数。

        num, name = tinytuple # 序列解包
        print(num, name) # 123 runoob

        list1 = ['Google', 'Taobao', 'Runoob', 'Baidu']
        tuple1 = tuple(list1)  # 将列表转为元组
        tupleWithList = ("aaa",list1)
        print("Taobao:", tupleWithList[1][1])

if __name__ == '__main__':
    元组()