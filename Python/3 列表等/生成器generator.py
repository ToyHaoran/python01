"""
列表受到内存限制，容量有限，且如果访问100万个元素的列表的前面几个，会浪费大量空间；
是否可以在循环的过程中不断推算出后续的元素呢？不必创建完整的list，从而节省大量的空间。
在Python中，这种一边循环一边计算的机制，称为生成器：generator。
"""
if __name__ == '__main__':
    g = (x**2 for x in range(10))
    print(next(g))
    for n in g:
        print(n, end=" ")
    print()

    # 如果一个函数定义中包含yield关键字，就变成了一个generator
    # 在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
    def yanghui_triangles(h):
        L, count = [1], 1  # count表示当前打印的层，h表示需要打印的高度
        while True:
            if count > h:
                return "Done"
            yield L
            count += 1
            L = [1] + [L[i] + L[i+1] for i in range(count-2)] + [1]


    f = yanghui_triangles(10)
    while True:
        try:
            print(next(f))
        except StopIteration as e:
            print(e.value)
            break
