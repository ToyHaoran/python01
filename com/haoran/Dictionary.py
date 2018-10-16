
def 字典():
    #————————————————————Dictionary（字典）又称Map
    # 列表是有序的对象集合，字典是无序的对象集合。两者之间的区别在于：字典当中的元素是通过键来存取的，而不是通过偏移存取。
    # 字典是一种映射类型，字典用"{ }"标识，它是一个无序的键(key) : 值(value)对集合。
    # 键(key)必须使用不可变类型(列表不行)。在同一个字典中，键(key)必须是唯一的。

    # dict与list相比：
    # 优点：
    # 1、查找和插入的速度极快，不会随着key的增加而变慢
        # 因为dict的实现原理和查字典是一样的。
        # 假设字典包含了1万个汉字，我们要查某一个字，一个办法是把字典从第一页往后翻，直到找到我们想要的字为止，这种方法就是在list中查找元素的方法，list越大，查找越慢
        # 第二种方法是先在字典的索引表里（比如部首表）查这个字对应的页码，然后直接翻到该页，找到这个字。无论找哪个字，这种查找速度都非常快，不会随着字典大小的增加而变慢。
    # 2、需要占用大量的内存，内存浪费多。
        # 是用空间来换取时间的一种方法

    if 0:
        dict1 = {}
        dict1['one'] = "1 - 菜鸟教程"
        dict1[2] = "2 - 菜鸟工具"
        print(dict1['one'])  # 输出键为 'one' 的值
        print(dict1[2])  # 输出键为 2 的值

        tinydict = {'name': 'runoob', 'code': 1, 'site': 'www.runoob.com'}
        print(tinydict)  # 输出完整的字典
        print(tinydict.keys())  # 输出所有键 # dict_keys(['name', 'code', 'site'])
        print(list(tinydict)) # ['name', 'code', 'site']
        print(tinydict.values())  # 输出所有值

        print("判断是否存在=======")
        print(tinydict.get("namexxx")) # 存在返回value，不存在默认返回None
        print("name" in tinydict)

        print("修改，删除字典===============")
        tinydict['code'] = 8  # 修改
        del tinydict['code']  # 删除字典中某个键
        # tinydict.clear() #清空字典
        # del tinydict #删除字典
        print(tinydict.pop("site")) # 删除字典中某个键,不存在报错
        print(tinydict)

        print("dict()直接从键-值对的序列构建字典==============")
        print(dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)]))
        print({x: x ** 2 for x in (2, 4, 6)})
        print(str(dict(Runoob=1, Google=2, Taobao=3)))

def 遍历技巧():
    if 0:
        print("遍历技巧=====================")
        knights = {'aaa': 'the a', 'bbb': 'the b'}
        # 在字典中遍历时，关键字和对应的值可以使用 items() 方法同时解读出来
        # 像scala的模式匹配
        for k, v in knights.items():
            print(k, v)

        list3 = [k + "=" + v for k, v in knights.items()] # list
        generator3 = (k + "=" + v for k, v in knights.items()) # generator
        print(list3) # ['aaa=the a', 'bbb=the b']
        print(type(list3)) # <class 'list'>

        # 在序列中遍历时，索引位置和对应值可以使用 enumerate() 函数同时得到
        for i, v in enumerate(['tic', 'tac', 'toe']):
            print(i, v) # 0 tic

        # 同时遍历两个或更多的序列，可以使用 zip() 组合
        questions = ['name', 'quest', 'favorite color']
        answers = ['lancelot', 'the holy grail', 'blue']
        for q, a in zip(questions, answers):
            print('%s：%s' % (q, a))
            print('{0}：{1}.'.format(q, a))

if __name__ == "__main__":
    字典()
    遍历技巧()