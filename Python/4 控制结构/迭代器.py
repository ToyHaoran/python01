from collections.abc import Iterable, Iterator

if __name__ == '__main__':
    # 判断一个对象是否可迭代，可直接作用于for循环
    print(isinstance([1, 2, 3], Iterable))
    # 判断是否是迭代器 list、dict、str不是，但可用iter()函数变换
    print(isinstance([1, 2, 3], Iterator))