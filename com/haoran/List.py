def 列表():
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
        # 详细的列表排序见 com/haoran/Function2.py 排序算法
        a.sort()  #[3, 3, 4, 9]
        a.clear()



def 堆栈():
    if 0:
        print("数据结构=======堆栈================")
        stack = [3, 4, 5]
        stack.append(6)
        stack.append(7) #[3, 4, 5, 6, 7]
        print(stack.pop())
        print(stack.pop())
        print(stack.pop())
        print(stack.pop())

def 队列():
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

def 列表推导式():
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

def 嵌套列表解析():
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

if __name__ == "__main__":
    列表()
    堆栈()
    队列()
    列表推导式()
    嵌套列表解析()
