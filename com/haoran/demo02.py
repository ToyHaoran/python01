#! /usr/bin/env python
# -*- coding: utf-8 -*-

# 斐波纳契数列
# 两个元素的总和确定了下一个数
if 0:
    print("斐波纳契数列=================")
    a, b = 0, 1
    while b < 10:
        print(b, end=",")
        a, b = b, a + b  # 先计算右边表达式，然后同时赋值给左边
    print()

if 0:
    print("while语句======================")
    # 在Python中没有do..while循环

    if 0:
        print("无限循环================")  # 无限循环在服务器上客户端的实时请求非常有用
        var = 1
        while var == 1:  # 表达式永远为 true
            num = int(input("输入一个数字 :"))
            if num==999:
                break
            print("你输入的数字是: ", num)
        print("Good bye!")

    print("while 循环使用 else 语句==============")
    count = 0
    while count < 5:
        print(count, " 小于 5")
        count = count + 1
    else: #条件为false时执行
        print(count, " 大于或等于 5")

if 1:
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

    print("Range==============")
    a = ['Google', 'Baidu', 'Runoob', 'Taobao', 'QQ']
    for i in range(len(a)):  # i：0、1、2、3、4
        print(i, a[i], end=",", sep=":")
    print()

    print("pass===============")
    for letter in 'Runoob':
        if letter == 'o':
            pass  # 如果没有内容，可以先写pass，占位。但是如果不写pass，就会语法错误
            #print('执行 pass 块')
        print('当前字母 :', letter)
    print("Good bye!")
