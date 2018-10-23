
语句_while = 0
if 0:
    print("while语句======================")
    # 在Python中没有do..while循环

    if 1:
        print("无限循环================")  # 无限循环在服务器上客户端的实时请求非常有用
        var = 1
        while var == 1:  # 表达式永远为 true
            num = int(input("输入一个数字 :"))
            if num == 999:
                break
            if num == 666:
                print("你中奖了")
                continue
            print("你输入的数字是: ", num)
        print("Good bye!")

    print("while 循环使用 else 语句==============")
    count = 0
    while count < 5:
        print(count, " 小于 5")
        count = count + 1
    else: # 条件为false时执行
        print(count, " 大于或等于 5")

语句_for_else = 0
if 0:
    #明确的知道循环执行的次数或者是要对一个容器进行迭代
    print("for语句及else语句=======================")
    sites = ["Baidu", "Google","Runoob","Taobao"]
    for site in sites:
        if site == "Runoob":
            print("菜鸟教程!")
            break
        print("循环数据 " + site)
    else: # 穷尽列表，没有一次进入循环时执行。
        print("sites为空，没有循环数据!")
    print("完成循环!")

语句_pass = 0
if 0:
    print("pass===============")
    for letter in 'Runoob':
        if letter == 'o':
            pass  # 如果没有内容，可以先写pass，占位。但是如果不写pass，就会语法错误
            #print('执行 pass 块')
        print('当前字母 :', letter)
    print("Good bye!")

    # 通常用于创建最小类
    class MyEmptyClass:
        pass

语句_range = 0
if 0:
    print("Range==============")
    # 如果你需要迭代一系列数字，内置函数 range()就派上用场了
    # range 永远不包括后面的数字
    for i in range(5, 20, 2): # 从5到20 增量为2
        print(i, end=";") # 5;7;9;11;13;15;17;19;
    print()

    print("要遍历序列的索引===")
    a = ['Google', 'Baidu', 'Runoob', 'Taobao', 'QQ']
    for i in range(len(a)):  # len为5，i为0、1、2、3、4
        print(i, a[i], end=",", sep=":")
    print()

    print(list(range(5)))# [0, 1, 2, 3, 4]
    print(list(reversed(range(5)))) # [4, 3, 2, 1, 0]
    print(list(enumerate(a))) #[(0, 'Google'), (1, 'Baidu'), (2, 'Runoob'), (3, 'Taobao'), (4, 'QQ')]

递归调用 = 0
if 0:
    # 递归函数的优点是定义简单，逻辑清晰
    # Python解释器没有对尾递归做优化。但是scala优化了。
    print("求n的阶乘=========")
    def fact(n):
        if n==1:
            return 1
        return n * fact(n - 1)
    print(fact(5))

循环小案例 = 0
if 0:
    # 斐波纳契数列
    # 两个元素的总和确定了下一个数
    print("斐波纳契数列=================")
    a, b = 0, 1
    while b < 10:
        print(b, end=",")
        a, b = b, a + b  # 先计算右边表达式，然后同时赋值给左边
    print()

if 0:
    """
   求解《百钱百鸡》问题
   1只公鸡5元 1只母鸡3元 3只小鸡1元 用100元买100只鸡
   问公鸡 母鸡 小鸡各有多少只
   """
    # 要理解程序背后的算法穷举法
    for x in range(0, 20):
        for y in range(0, 33):
            z = 100 - x - y
            if 5 * x + 3 * y + z / 3 == 100:
                print('公鸡: %d只, 母鸡: %d只, 小鸡: %d只' % (x, y, z))
            else:
                print("dd")

if 0:
    """
    找出100~999之间的所有水仙花数
    水仙花数是各位立方和等于这个数本身的数
    如: 153 = 1**3 + 5**3 + 3**3
    """
    for num in range(100, 1000):
        low = num % 10
        mid = num // 10 % 10
        high = num // 100
        if num == low ** 3 + mid ** 3 + high ** 3:
            print(num)

if 0:
    """
    输出2~99之间的素数
    """
    import math
    for num in range(2, 100):
        is_prime = True
        for factor in range(2, int(math.sqrt(num)) + 1):
            if num % factor == 0:
                is_prime = False
                break
        if is_prime:
            print(num, end=' ')

if 0:
    """
    输出乘法口诀表(九九表)
    """
    for i in range(1, 10):
        for j in range(1, i + 1):
            print('%d*%d=%d' % (i, j, i * j), end='\t')
        print()

生成器 = 0
if 0:
    print("生成器====yield====================")


    def fibonacci(n, w=0):  # 生成器函数 - 斐波那契
        a, b, counter = 0, 1, 0
        while True:
            if (counter > n):
                return
            yield a
            a, b = b, a + b
            print('%d,%d' % (a, b))
            counter += 1


    # 如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator
    # f 是一个迭代器，由生成器返回生成
    f = fibonacci(10, 0)
    while True:
        try:
            print(next(f), end=" ")
        except StopIteration:
            break

生成器小案例 = 0
if 0:
    print("生成器小案例============")
    # 打个比方的话，yield有点像断点。
    # 加了yield的函数，每次执行到有yield的时候，会返回yield后面的值
    # 并且函数会暂停，直到下次调用或迭代终止；
    def myYield_1():
        a, i = 'yield', 0
        while True:
            print('before %d' % i, end=", ")
            yield a, i # 暂停
            print('after %d' % i, end=", ")
            i += 1

    def myYield_2():
        a, i = 'yield_a', 0
        b, i = 'yield_b', 0
        while True:
            print('before %d' % i, end=", ")
            yield a, i # 暂停
            yield b, i # 暂停
            print('after %d' % i, end=", ")
            i += 1

    it1 = iter(myYield_1())
    it2 = iter(myYield_2())

    for i in range(10):
        print("next %d" % i, end=": ")
        print(next(it1))
    print('\n')

    for i in range(10):
        print("next %d" % i, end=": ")
        print(next(it2))

迭代器 = 0
if 0:
    print("迭代器=====================")
    # 我们已经知道，可以直接作用于for循环的数据类型有以下几种：
    # 一类是集合数据类型，如list、tuple、dict、set、str等；
    # 一类是generator，包括生成器和带yield的generator function。
    # 这些可以直接作用于for循环的对象统称为可迭代对象：Iterable。
    from _collections_abc import Iterable
    print(isinstance([], Iterable)) # True
    # 可以被next()函数调用并不断返回下一个值的对象称为迭代器：Iterator
    from _collections_abc import Iterator
    print(isinstance([], Iterator)) # False
    # 生成器都是Iterator对象，但list、dict、str虽然是Iterable，却不是Iterator。
    # 把list、dict、str等Iterable变成Iterator可以使用iter()函数：
    list1 = [1, 2, 3, 4]
    it = iter(list1)
    print(next(it))  # 1
    print(next(it))  # 2
    for x in it:
        print(x, end=" ")  # 3,4
    print()

    print("自定义迭代器=======================")
    class MyNumbers:
        def __iter__(self):
            self.a = 1
            return self

        def __next__(self):
            if self.a <= 10:
                x = self.a
                self.a += 1
                return x
            else:
                raise StopIteration

    myclass = MyNumbers()
    myiter = iter(myclass)
    for x in myiter:
        print(x)
        # print(next(myiter))#抛出异常
