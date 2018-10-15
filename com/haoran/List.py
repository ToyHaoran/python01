def 列表():
    if 1:
        # 参考：
        # https://docs.python.org/3.7/tutorial/introduction.html
        # https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014316724772904521142196b74a3f8abf93d8e97c6ee6000
        # http://www.runoob.com/python3/python3-module.html
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

        print("list基本操作==============")
        del a[2]  #[9, 2] #基于索引删除
        a.append(3)
        a.append([3,4,5])  #[9, 2, 3, [3, 4, 5]]
        a.extend([3,4,5])  #[9, 2, 3, [3, 4, 5], 3, 4, 5]
        print(a.count(3)) #2
        print(a.index(3))# 2
        a.insert(3,666)  #[9, 2, 3, 666, [3, 4, 5], 3, 4, 5]
        print(a.pop())#5 默认移除最后一个元素
        print(a.pop(1))#2 移除下标为1的元素
        a.remove(666)  #[9, 3, [3, 4, 5], 3, 4]
        a.reverse()#反转
        a.pop(2)
        a.sort()  #[3, 3, 4, 9]


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
        #在列表的最后添加或者弹出元素速度快，
        # 然而在列表里插入或者从头部弹出速度却不快（因为所有其他的元素都得一个一个地移动）
        from collections import deque
        queue = deque(["aa", "bb", "cc"])
        queue.append("dd")           # Terry arrives
        queue.append("ee")          # Graham arrives
        print(queue.popleft())
        print(queue.popleft())
        print(queue.popleft())

def 列表推导式():
    if 0:
        print("列表推导式==============")
        vec1 = [2, 4, 6]
        vec2 = [4, 3, -9]
        print([3 * x for x in vec1]) #[6, 12, 18]
        print([3*x for x in vec1 if x > 3]) #[12, 18]
        print([x*y for x in vec1 for y in vec2]) #[8, 6, -18, 16, 12, -36, 24, 18, -54]
        print([vec1[i]*vec2[i] for i in range(len(vec1))]) #[8, 12, -54]
        print([str(round(355/113, i)) for i in range(1, 6)]) #['3.1', '3.14', '3.142', '3.1416', '3.14159']

def 嵌套列表解析():
    if 0:
        print("嵌套列表解析===============")
        matrix = [[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],]
        #将3X4的矩阵列表转换为4X3列表(后面for先执行)
        print([[row[i] for row in matrix] for i in range(4)])
        #或者
        transposed = []
        for i in range(4): transposed.append([row[i] for row in matrix])

if __name__ == "__main__":
    列表()
    堆栈()
    队列()
    列表推导式()
    嵌套列表解析()
