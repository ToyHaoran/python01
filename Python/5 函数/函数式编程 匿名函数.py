
if __name__ == '__main1__':
    print("变量可以指向函数，函数名也是变量(被占用)========")
    f = abs  # abs = 20
    print("函数作为参数=========")

    def add(x, y, f):
        return f(x) + f(y)
    print(add(-5, 6, f))

if __name__ == '__main__':
    print("map reduce======================")
    # Lambda表达式(匿名函数)：只能有一个表达式，不用写return，返回值就是该表达式的结果。
    # map将传入的函数依次作用到序列的每个元素，并把结果作为新的Iterator返回；惰性序列
    iter1 = map(lambda x: x*x, [1, 2, 3, 4, 5])
    print(list(iter1))

    #  reduce对所有元素进行折叠，把结果继续和序列的下一个元素做累积计算
    from functools import reduce
    sum = reduce(lambda x, y: x+y, list(range(10)))
    print(sum)  # 45 返回一个数

    # filter() 同map，保留为True的元素；
    iter2 = filter(lambda x: x%2==1, [1, 2, 4, 5, 6, 9, 10, 15])  # 过滤奇数
    print(list(iter2))  # [1, 5, 9, 15]
