
if __name__ == '__main1__':
    # for循环 对一个容器进行迭代
    sites = ["Baidu", "Google", "toy", "Taobao"]
    for site in sites:
        pass
    else:  # 穷尽列表，没有一次进入循环时执行。
        print("sites为空，没有循环数据!")

    # Range 迭代某一范围的数字(左闭右开)
    for i in range(5, 20, 1):  # 从5到19 增量为2
        pass

    # 下标循环
    for i, site in enumerate(sites):
        pass

if __name__ == '__main__':
    # while语句 在Python中没有do..while循环
    while True:
        num = int(input("输入一个数字 :"))
        print("你输入的数字是: ", num)
        if num == 999:
            break
        if num == 666:
            print("你中奖了")
            continue
    print("Good bye!")
