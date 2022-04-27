"""
字典是一种映射类型，字典用"{ }"标识，它是一个无序的键(key) : 值(value)对集合。
键(key)必须使用不可变类型(列表不行)。在同一个字典中，键(key)必须是唯一的。

dict与list相比：
字典是无序的对象集合，列表是有序的对象集合。
字典当中的元素是通过键来存取的，而不是通过偏移存取。
字典查找和插入的速度极快，不会随着key的增加而变慢，实现原理和查字典是一样的。
字典需要占用大量的内存，内存浪费多，是用空间来换取时间的一种方法
"""

if __name__ == '__main__':
    dict1 = {'one': "教程", 2: "工具"}  # key必须是不可变对象
    dict1[7] = "插入"
    print(dict1['one'], dict1[2])  # 输出键为 'one'和2 的值
    print(dict1.keys(), dict1.values())  # 输出所有键，所有值
    print(dict1.get("one"))  # 判读key=one是否存在 存在返回value，不存在默认返回None
    print("one" in dict1)  # 判读key=one是否存在 True
    print(dict1.pop('one'))  # 删除字典中某个键,不存在报错
    dict1.update({'b': 'foo', 'c': 12})  # 字典连接

    # 字典遍历
    for k, v in dict1.items():
        print(k, v)
