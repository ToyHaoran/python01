
def 集合():
    if 1:
        # 集合（set）是一个无序不重复元素的序列。
        # 可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典
        student = {'Tom', 'Jim', 'Mary', 'Tom', 'Jack', 'Rose'}
        print(student) # 输出集合，重复的元素被自动去掉
        # 成员测试
        print('Rose' in student)

        print("添加，删除=============")
        student.add("zhangshan")
        student.update({1, 3}, [4, 5])
        print(student)  # {1, 'Mary', 3, 4, 5, 'Rose', 'Jack', 'Tom', 'Jim', 'zhangshan'}

        student.remove(1)  # 不存在此元素会报错
        student.discard(3)  # 不存在不会报错
        print(student)  # {'Mary', 4, 5, 'Rose', 'Jack', 'Tom', 'Jim', 'zhangshan'}

        print("集合运算==============")
        a = set('abracadabra')
        b = set('alacazam')
        print(a) #去除重复字母
        print(b)
        print(a - b) # a和b的差集
        print(a | b) # a和b的并集
        print(a & b) # a和b的交集
        print(a ^ b) # a和b中不同时存在的元素

        print("推导式========")
        a = {x for x in 'abradcradabra' if x not in 'abc'}
        print(a) #{'d', 'r'}


if __name__ == '__main__':
    集合()