
数据类型转换 = 0
if 0:
    print("数据类型转换===========")
    print(int('123'))
    print(int(12.34))
    print(float('12.34'))
    print(str(1.23))
    print(str(100))
    print(bool(1))
    print(bool(''))

return语句 = 0
if 0:
    print("return语句===========")
    def fun(a, b):
        "返回多个值，结果以元组形式表示"
        return a, b, a + b
    print(fun(1, 2))

传递不可变对象 = 0
if 0:
    print("传递不可变对象=======")
    def changeInt(a):
        a = 10
    b = 2
    changeInt(b)
    print(b)  # 结果是 2

传递可变对象 = 0
if 0:
    print("传递可变对象=========")
    def changList(list):
        list.append([3,4,5])
    list1 = [1,2]
    changList(list1)
    print(list1)

默认参数 = 0
if 0:
    print("默认参数============")
    def printinfo(name, age=35):
        print("名字: ", name, end=";  ")
        print("年龄: ", age)
        return
    printinfo(age=50, name="runoob")
    printinfo(name="runoob")

# 定义默认参数要牢记一点：默认参数必须指向不变对象！
# 看下面两个例子
if 0:
    # 默认值在定义范围内的函数定义点进行计算,默认值仅计算一次
    i = 5
    def f(arg=i):
        print(arg)
    i = 6
    f() # 5

if 0:
    # 巨坑
    # 当默认值是可变对象（例如列表，字典或大多数类的实例）时，会累积在后续调用中传递给它的参数
    def f(a, L=[]):
        L.append(a)
        return L
    print(f(1))
    print(f(2)) # [1, 2]
    print(f(3)) # [1, 2, 3]

    # 如果不想累计，解决方法：
    def f2(a, L=None):
        if L is None:
            L = []
        L.append(a)
        return L
    print(f2(1))
    print(f2(2))
    print(f2(3))

不定长参数 = 0
if 0:
    print("不定长参数===*元组=============")
    def printStrs(aa, *strs):
        print(aa)
        print(strs)
    def priStrs(*strs, aa):
        print(aa)
        print(strs)
    printStrs("aaa", "bbb", "ccc")
    priStrs("bbb", "ccc", aa="aaa") #在不定长参数后面的参数必须是关键字参数，否则没法匹配

if 0:
    print("解压缩参数列表=======")
    args = [3, 7]
    print(list(range(*args))) # [3, 4, 5, 6]


关键字参数 = 0
if 0:
    print("不定长参数===**字典=========")
    def printStrs2(**strs):
        print(strs)
    # printStrs2("aa","bb") 报错，必须是字典格式
    printStrs2(a=1, b=2)

命名关键字参数 = 0
if 0:
    # 必须要加：特殊分隔符*
    # *后面的关键字参数
    def person(name, age, *, city, job):
        print(name, age, city, job)
    #调用
    person("jack", 20, city="hainan", job="xxx")

参数组合 = 0
if 0:
    # 虽然可以组合多达5种参数，但不要同时使用太多的组合，否则函数接口的可理解性很差。
    print("参数组合===========")
    def f1(a, b, c=0, *args, **kw):
        print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)
    def f2(a, b, c=0, *, d, **kw):
        print('a =', a, 'b =', b, 'c =', c, 'd =', d, 'kw =', kw)

    f1(1, 2)
    f1(1, 2, c=3)
    f1(1, 2, 3, 'a', 'b')
    f1(1, 2, 3, 'a', 'b', x=99, y=100)
    f2(1, 2, d=99, ext=None)

    args = (1, 2, 3, 4)
    kw = {'d': 99, 'x': '#'}
    f1(*args, **kw) # 解压

变量作用域 = 0
if 0:
    print("变量作用域======================")
    #从内往外查找变量
    built_in_count = int(2.9)  # 内建作用域
    global_count = 0  # 全局作用域
    def outer():
        enclosing_count = 1  # 闭包函数外的函数中
        def inner():
            local_count = 2  # 局部作用域
    #Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，
    # 其它的代码块（如 if/elif/else/、try/except、for/while等）是不会引入新的作用域的，
    # 也就是说这些语句内定义的变量，外部也可以访问
    if 1:
        msg = "hello"
    print(msg)

if 0:
    print("global和nonlocal关键字==================")
    #当内部作用域想修改外部作用域的变量时，就要用到global和nonlocal关键字了
    num = 1  # 这个是不能放在方法内部的。
    def fun1():
        global num # 需要使用 global 关键字声明
        num = 123
    print(num)
    fun1()
    print(num)

if 0:
    #修改嵌套作用域（enclosing 作用域，外层非全局作用域）中的变量则需要 nonlocal 关键字
    def outer():
        num = 10
        def inner():
            nonlocal num # nonlocal关键字声明
            num = 100
        print(num)
        inner()
        print(num)
    outer()

