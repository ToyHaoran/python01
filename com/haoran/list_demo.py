
列表 = 0
if 0:
    print("列表====================================")
    #列表可以完成大多数集合类的数据结构实现。列表中元素的类型可以不相同，它支持数字，字符串甚至可以包含列表（所谓嵌套）
    print("获取元素========================")
    list = ['abcd', 786, 2.23, 'runoob', 70.2]
    tinylist = [123, 'runoob']
    print(list)  # 输出完整列表 ['abcd', 786, 2.23, 'runoob', 70.2]
    print(list[:]) #同上
    print(list[0])  # 输出列表第一个元素 abcd
    print(list[1:3])  # 从第二个开始输出到第三个元素（左毕右开） [786, 2.23]
    print(list[2:])  # 输出从第三个元素开始的所有元素 [2.23, 'runoob', 70.2]
    print(tinylist * 2)  # 输出两次列表 [123, 'runoob', 123, 'runoob']
    print(list + tinylist)  # 连接列表 ['abcd', 786, 2.23, 'runoob', 70.2, 123, 'runoob']
    print(len(list))
    for x in list: print(x,end=" ")
    print()

    print("改变list元素=============")
    a = [1, 2, 3, 4, 5, 6]; a[0] = 9; a[2:5] = [13, 14, 15] #[9, 2, 13, 14, 15, 6]
    print(max(a))
    print(min(a))
    #print(list((1,2,3,4)))#将元组转为list  (注意这里名字被占用了。。。)
    a[2:5] = [] # 将对应的元素值设置为 []   # [9, 2, 6]


    print("list基本方法==============")
    del a[2]  #[9, 2] #基于索引删除元素
    a.append(3)
    a.append([3,4,5])  # [9, 2, 3, [3, 4, 5]]
    a.extend([3,4,5])  # [9, 2, 3, [3, 4, 5], 3, 4, 5] # 迭代元素
    print(a.index(3)) # 2
    print(a.index(3, 3)) # 4 #从第三个位置开始查找
    a.insert(3,666)  #[9, 2, 3, 666, [3, 4, 5], 3, 4, 5]
    print(a.pop())#5 默认移除最后一个元素
    print(a.pop(1))#2 移除下标为1的元素
    a.remove(666)  #[9, 3, [3, 4, 5], 3, 4]
    a.reverse()#反转
    a.pop(2)
    print(a.count(3)) # 3出现的次数
    print(len(a)) # a的长度是多少
    # 详细的列表排序见 com/haoran/function_demo2.py 排序算法
    a.sort()  #[3, 3, 4, 9]
    a.clear()

列表去重的几种方式 = 0
if 0:
    print("使用内置set方法来去重==常用====")
    lst1 = [2, 1, 3, 4, 1]
    lst2 = list(set(lst1))
    print(lst2)

    print("使用字典中fromkeys()的方法来去重====")
    lst1 = [2, 1, 3, 4, 1]
    lst2 = {}.fromkeys(lst1).keys()
    print(lst2)

    print("使用常规方法来去重=====")
    lst1 = [2, 1, 3, 4, 1]
    temp = []
    for item in lst1:
        if not item in temp:
            temp.append(item)
    print(temp)

    print("使用列表推导来去重=======")
    lst1 = [2, 1, 3, 4, 1]
    temp = []
    [temp.append(i) for i in lst1 if not i in temp]
    print(temp)

    print("使用sorted函数来去重=====")
    lst1 = [2, 1, 3, 4, 1]
    # 能保证顺序
    lst2 = sorted(set(lst1), key=lst1.index)
    print(lst2)

统计list中各个元素出现的次数 = 0
# https://blog.csdn.net/sinat_24091225/article/details/77925473
if 0:
    print("利用字典dict来完成统计==（较慢）=====")
    a = [1, 2, 3, 1, 1, 2]
    dict1 = {}
    for key in a:
        dict1[key] = dict1.get(key, 0) + 1
    print(dict1)

    print("利用Python的collection包下Counter的类===（特别快）====")
    from collections import Counter
    a = [1, 2, 3, 1, 1, 2]
    print(dict(Counter(a)))

    print("numpy包中的unique======（最快）=======")
    import numpy as np
    lst = [1, 2, 3, 1, 1, 2]
    print(dict(zip(*np.unique(lst, return_counts=True))))

    print("pandas包下的value_counts方法=========")
    import pandas as pd
    a = [1, 2, 3, 1, 1, 2]
    result = pd.value_counts(a)
    print(result)

    print("矩阵====")
    a = pd.DataFrame([[1, 2, 3],
                      [3, 1, 3],
                      [1, 2, 1]])
    result = a.apply(pd.value_counts)
    print(result)
    #     0    1    2
    # 1  2.0  1.0  1.0  # 表示元素1在第一列出现2次，在第二列出现1次，在第三列出现1次
    # 2  NaN  2.0  NaN  # 表示元素2在第一列出现0次，在第二列出现2次，在第三列出现0次
    # 3  1.0  NaN  2.0  # 表示元素3在第一列出现1次，在第二列出现0次，在第三列出现2次

数据结构_堆栈 = 0
if 0:
    print("数据结构=======堆栈================")
    stack = [3, 4, 5]
    stack.append(6)
    stack.append(7) #[3, 4, 5, 6, 7]
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())

数据结构_队列 = 0
if 0:
    print("数据结构=======队列================")
    # 在列表的最后添加或者弹出元素速度快，
    # 然而在列表里插入或者从头部弹出速度却不快（因为所有其他的元素都得一个一个地移动）
    from collections import deque
    queue = deque(["aa", "bb", "cc"])
    queue.append("dd")
    queue.append("ee")
    print(queue.popleft())
    print(queue.popleft())
    print(queue.popleft())

列表推导式 = 0
if 0:
    print("列表推导式==============")
    vec1 = [2, 4, 6]
    vec2 = [4, 3, -9]
    print([3 * x for x in vec1])  # [6, 12, 18]
    print([3 * x for x in vec1 if x > 3])  # [12, 18]
    print([(x, y, x * y) for x in vec1 for y in vec2])  # [(2, 4, 8), (2, 3, 6), (2, -9, -18), (4, 4, 16), (4, 3, 12), (4, -9, -36), (6, 4, 24), (6, 3, 18), (6, -9, -54)]
    print([vec1[i] * vec2[i] for i in range(len(vec1))])  # [8, 12, -54]
    from math import pi
    print([str(round(pi, i)) for i in range(1, 6)])  # ['3.1', '3.14', '3.142', '3.1416', '3.14159']

    print(list(map(lambda x: x**2, range(10))))
    print([x**2 for x in range(10)]) #和上面等价

嵌套列表解析 = 0
if 0:
    print("嵌套列表解析===============")
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],]
    # 将3X4的矩阵列表转换为4X3列表(后面for先执行)
    print([[row[i] for row in matrix] for i in range(4)]) # [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]
    # 或者等价于
    transposed = []
    for i in range(4):
        transposed.append([row[i] for row in matrix])
    print(transposed)
    # 或者等价于
    transposed = []
    for i in range(4):
        transposed_row = []
        for row in matrix:
            transposed_row.append(row[i])
        transposed.append(transposed_row)
    print(transposed)

    # 或者(注意是元组)
    print("zip==========") # Make an iterator that aggregates elements from each of the iterables.
    # 返回元组的迭代器，其中第i个元组包含来自每个参数序列或迭代的第i个元素。当最短输入可迭代用尽时，迭代器停止
    print(list(zip(*matrix))) # [(1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12)]
    print(list(zip([1, 2, 3, 4], [5, 6]))) # [(1, 5), (2, 6)]
    import itertools # 为高效循环创建迭代器的函数
    print(list(itertools.zip_longest([1, 2, 3, 4], [5, 6]))) # [(1, 5), (2, 6), (3, None), (4, None)]

