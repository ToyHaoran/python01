#————————————String（字符串）
if 0:
    # ——————————字符串相关操作
    word = "I am "
    name = 'lihaoran'
    # """实现多行字符串
    # 三引号，常用语引用HTML代码，或SQL代码
    # 在Python3中，所有的字符串都是Unicode字符串。
    helloword = ("""\
        hello
        world""")
    print(helloword)

    # 如果代码太长写成一行不便于阅读 可以使用\或()折行
    str1 = word + \
          name
    print(str1)

    print("转义字符============")
    print("aaa\\bbb\'ccc\"ddd\teee\nfff")

    print("不使用转义字符(r表示原始字符串，不发生转义)=========")
    print(r"xxx\nlll")

    print("字符串拼接==============")
    a = "hello"
    b = "world"
    c = 21
    print(a + b)#helloword  注意+号连接大量字符串是非常低效的。
    print(a,c)#hello 21 默认以空格分隔
    #print(a + c)# 错误，不能将int转为String
    print(a + str(c))# 如果报错str不可用，说明你把内置的函数给占用了。

if 0:
    print("字符串访问==========")
    #索引值以 0 为开始值，-1 为从末尾的开始位置。
    #从后面索引：-6 -5 -4 -3 -2 -1
    #从前面索引： 0  1  2  3  4  5
    #            a  b  c  d  e  f
    #从后面截取：  1  2  3  4  5
    #从前面截取： -5 -4 -3 -2 -1
    str = 'Runoob'
    print(str)  # 输出字符串
    print(str[-1])  # 输出倒数第一个字符 b
    print(str[0])  # 输出字符串第一个字符 R
    print("注意，切片总是左毕右开原则")
    print(str[2:5])  # noo
    print(str[:3] + " " + str[3:]) #Run oob
    print(str * 2)  # 输出字符串两次 RunoobRunoob
    # Python 没有单独的字符类型，一个字符就是长度为1的字符串。
    print(str[0] == str[0:1]) # True
    word = 'Python'
    # 检查是否是子字符串
    print('P' in word)


if 0:
    print("字符串格式化=============")
    print("我叫 %s 今年 %d 岁!" % ('小明', 10))#我叫 小明 今年 10 岁!
    print( "{1} {0} {1}".format("hello", "world"))
    # 通过字典设置参数
    site = {"name": "菜鸟教程", "url": "www.runoob.com"}
    print("网站名：{name}, 地址 {url}".format(**site))
    # 通过列表索引设置参数
    my_list = ['菜鸟教程', 'www.runoob.com']
    print("网站名：{0[0]}, 地址 {0[1]}".format(my_list)) # "0" 是必须的
    print("数字格式化================")
    print("百分号的用法，比较旧了，会逐渐淘汰掉========")
    #百分号的用法%[(name)][flags][width].[precision]typecode
    print("%6.3f" % 2.3)#宽度为6，小数3位，右对齐，浮点型，前面有一空格 2.300
    print("%.2f" % 2.232)#保留两位小数
    print("%+5x" % -10)#右对齐，宽度5，前面3空格
    import math
    print ("pi的值是%s" % math.pi)
    print("%10.*f" % (4, 1.2))#    1.2000
    print("format的用法=========================")
    print("{:.2f}".format(3.1415926))#保留两位小数
    print("{:+.2f}".format(-3.1415926))#带符号保留两位小数
    print("{:.0f}".format(3.14))#不保留小数
    print("{:0>2d}".format(5))#数组补零，填充左边，宽度为2
    print("{:x<4d}".format(10))#数字补x，填充右边，宽度为4
    print("{:,}".format(1000000))#逗号分隔
    print("{:.2%}".format(0.25))#百分比格式，两位小数
    print("{:.2e}".format(1000000))#指数记法 1.00e+06
    print("{:10d}".format(13))#右对齐，保持10位
    print("{:>10d}".format(13))#右对齐，保持10位
    print("{:<10d}".format(13))#左对齐
    print("{:^10d}".format(13))#居中对齐
    print('{:b}'.format(11))#二进制
    print('{:d}'.format(11))#十进制
    print('{:o}'.format(11))#八进制
    print('{:x}'.format(11))#十六进制 b
    print('{:#x}'.format(11))#0xb
    print('{:#X}'.format(11))#0XB

if 1:
    print("字符串内建函数====================")
    str = "hello"
    str2 = "123"
    print(str.capitalize()) # 首字母大写
    print(str.center(9,"-"))
    print(str.count("ll",0,5)) # 字母出现的次数
    print(str.encode().decode())
    print(str.startswith("he",0,10)) #字符串是不是以he开头
    print(str.endswith("lo",0,10)) #字符串是不是以lo结尾
    print(str.find("e")) # 1
    print(str.rfind("e")) # 1
    print(str.index("e"))
    print(str.rindex("e"))
    print((str+str2).isalnum()) # 如果字符串中的所有字符都是字母数字且至少有一个字符，则返回true，否则返回false
    print(str2.isdigit())
    print(str.isalpha())
    print(str.islower())
    print(str.isupper()) # 字符串是不是大写
    print(str.istitle())
    print(len(str))
    str3 ="-"
    print(str3.join(("aaa","bbb","ccc")))
    print(str.lower())
    print(str.title()) #单词首字母大写
    print(str.upper()) #字符串变大写
    print(str.swapcase())
    print(str.ljust(10,"-"))
    print(str.rjust(10,"-"))
    print(str.lstrip("he"))
    print(str.rstrip("lo"))
    print(str.strip("o"))
    print(max(str))
    print(min(str))
    print(str.replace("he","HE",1))
    print(str.split("e"))
