if __name__ == '__main__':
    #标准库概览
    if 0:
        print("操作系统接口=============")
        import os
        print("返回当前的工作目录===============")
        print(os.getcwd())
        print("修改当前的工作目录=============")
        os.chdir("../test01")
        print("执行系统命令:和你在cmd中敲的命令是一样的===============")
        os.system("dir")
        print(dir(os))

    if 0:
        import shutil
        print("日常的文件和目录管理任务============")
        #shutil.copyfile("demoxxxx.py", "../test01/xxxxx.py")

    if 0:
        print("文件通配符=====================")
        #glob模块提供了一个函数用于从目录通配符搜索中生成文件列表:
        import glob
        print(glob.glob("*.py")) #打印目录下所有py结尾的文件。

    if 0:
        print("命令行参数=============")
        #通用工具脚本经常调用命令行参数。这些命令行参数以链表形式存储于 sys 模块的 argv 变量。
        # 例如在命令行中执行 "python demo.py one two three" 后可以得到以下输出结果:
        import sys
        print(sys.argv) #就是运行时在python.exe后面的代码 H:/code/idea/python/com/haoran/module_overview.py

    if 0:
        print("错误输出重定向和程序终止=================")
        import sys
        #sys 还有 stdin，stdout 和 stderr 属性，即使在 stdout 被重定向时，后者也可以用于显示警告和错误信息。
        sys.stderr.write('Warning, log file not found starting a new one\n')
        #大多脚本的定向终止都使用 "sys.exit()"。

    if 0:
        print("字符串正则匹配==================")
        # re模块为高级字符串处理提供了正则表达式工具。对于复杂的匹配和处理，正则表达式提供了简洁、优化的解决方案:
        import re
        print(re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest'))
        print(re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat'))
        #如果只需要简单的功能，应该首先考虑字符串方法，因为它们非常简单，易于阅读和调试:
        print('tea for too'.replace('too', 'two'))

    if 0:
        print("数学=================")
        import math
        print(math.cos(math.pi / 4))
        print(math.log(1024, 2))

        import random
        print(random.choice(['apple', 'pear', 'banana'])) #随机选一个
        print(random.sample(range(100), 10)) # 从100个中选10个
        print(random.random()) #一个随机float
        print(random.randrange(6)) #random integer chosen from range(6)

    if 0:
        print("访问互联网==============(有点问题，无法实现，跳过)")
        # 有几个模块用于访问互联网以及处理网络通信协议。
        # 其中最简单的两个是用于处理从 urls 接收的数据的 urllib.request 以及用于发送电子邮件的 smtplib:
        from urllib.request import urlopen
        for line in urlopen('http://tycho.usno.navy.mil/cgi-bin/timer.pl'):
            line = line.decode('utf-8')  # Decoding the binary data to text.
            if 'EST' in line or 'EDT' in line:  # look for Eastern Time
                print(line)

    if 0:
        print("日期和时间===============")
        # datetime模块为日期和时间处理同时提供了简单和复杂的方法。
        # 支持日期和时间算法的同时，实现的重点放在更有效的处理和格式化输出。
        # 该模块还支持时区处理:
        from datetime import date
        now = date.today()
        print(now)
        print(now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))
        birthday = date(1996, 5, 23)
        age = now - birthday
        print(age.days)
    
    if 0:
        print("数据压缩=============")
        #以下模块直接支持通用的数据打包和压缩格式：zlib，gzip，bz2，zipfile，以及 tarfile。
        import zlib
        s = b'witch which has which witches wrist watch'
        print(len(s))
        t = zlib.compress(s)
        print(t)
        print(len(t))
        print(zlib.decompress(t))
        print(zlib.crc32(s))

    if 0:
        print("性能度量============")
        # 有些用户对了解解决同一问题的不同方法之间的性能差异很感兴趣。Python 提供了一个度量工具，为这些问题提供了直接答案。
        # 例如，使用元组封装和拆封来交换元素看起来要比使用传统的方法要诱人的多,timeit 证明了现代的方法更快一些。
        from timeit import Timer
        print(Timer('t=a; a=b; b=t', 'a=1; b=2').timeit())
        print(Timer('a,b = b,a', 'a=1; b=2').timeit())
        #相对于 timeit 的细粒度，:mod:profile 和 pstats 模块提供了针对更大代码块的时间度量工具。

    if 0:
        print("测试模块==============没看懂，跳过")
        # 开发高质量软件的方法之一是为每一个函数开发测试代码，并且在开发过程中经常进行测试
        # doctest模块提供了一个工具，扫描模块并根据程序中内嵌的文档字符串执行测试。
        # 测试构造如同简单的将它的输出结果剪切并粘贴到文档字符串中。
        # 通过用户提供的例子，它强化了文档，允许 doctest 模块确认代码的结果是否与文档一致:
        def average(values):
            return sum(values) / len(values)
        import doctest
        doctest.testmod()   # 自动验证嵌入测试
        print(average([20, 30, 70]))












