"""
集合(set)是一个无序不重复元素的序列。
"""

if __name__ == '__main__':
    nullset = set()  # 创建空集合 {}创建空字典
    student = {'Tom', 'Jim', 'Mary', 'Tom', 'Jack', 'Rose'}  # 自动去重
    print('Rose' in student)  # 成员测试
    student.add("zhangshan")  # 添加
    student.update({1, 3}, [4, 5])
    print(student)  # {1, 'Mary', 3, 4, 5, 'Rose', 'Jack', 'Tom', 'Jim', 'zhangshan'}
    student.remove(1)  # 删除元素，不存在此元素会报错
    student.discard(3)  # 不存在不会报错
    print(student)  # {'Mary', 4, 5, 'Rose', 'Jack', 'Tom', 'Jim', 'zhangshan'}

    print("集合运算==============")
    a, b = set('abracadabra'), set('alacazam')
    print(a - b)  # a和b的差集
    print(a | b)  # a和b的并集
    print(a & b)  # a和b的交集
    print(a ^ b)  # a和b中不同时存在的元素
