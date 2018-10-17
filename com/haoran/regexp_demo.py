
def 正则表达式():
    import re
    print("正则表达式==================")
    # re模块为高级字符串处理提供了正则表达式工具。对于复杂的匹配和处理，正则表达式提供了简洁、优化的解决方案:
    # 如果只需要简单的功能，应该首先考虑字符串方法，因为它们非常简单，易于阅读和调试:
    # print('tea for too'.replace('too', 'two'))
    if 0:
        a = 'uav,ubv ucv,uwv   UZV,ucv,UOV  123-4-6-789'
        print("findall==========")
        print(re.findall('u[^abc]v', a)) #取u和v中间不是a或b或c的字符
        print(re.findall("\d", a)) #匹配数字
        print("split=============")
        print(re.split('\s+', a)) #通过空格切分
        print("finditer==========")
        m = re.finditer("\d", a)
        for iter in m:
            print(iter) #每一个都是一个match对象  <re.Match object; span=(39, 40), match='7'>
            print(iter.group())


    if 0:
        print("sub=============")
        #替换字符串中的字符，这时候就可以用到 def sub(pattern, repl, string, count=0, flags=0)
        print(re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat'))
        a = 'Python*Android*Java-888'
        if 0:
            # 把字符串中的 * 字符替换成 & 字符
            sub1 = re.sub('\*', '&', a)
            print(sub1)
        if 0:
            # 把字符串中的第一个 * 字符替换成 & 字符
            sub2 = re.sub('\*', '&', a, 1)
            print(sub2)
        if 1:
            # 把字符串中的 * 字符替换成 & 字符,把字符 - 换成 |
            # 1、先定义一个函数
            def convert(value):
                group = value.group()
                if (group == '*'):
                    return '&'
                elif (group == '-'):
                    return '|'
            # 第二个参数，要替换的字符可以为一个函数
            sub3 = re.sub('[\*-]', convert, a)
            print(sub3)

    if 0:
        print("match===========")
        pattern = re.compile(r'hello')
        match = pattern.match("hello world!")
        if match:
            print(match)
            print(match.string)
            print(match.re)
            print(match.group())
            print(match.span())
            print(match.start())
            print(match.end())

        print("简化上面代码============")
        # re.match(pattern, string, flags=0)
        # re.match 只匹配开始，相当于starwith，如果起始位置匹配失败，返回none。
        a = "hello world!"
        print(re.match('hello', a))  # 在起始位置匹配 <re.Match object; span=(0, 5), match='hello'>
        print(re.match('world', a))  # 不在起始位置匹配 None

        print("匹配组===============")
        m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello world!')
        print(m.groups()) #('hello', 'world', '!')
        print(m.group(1,2)) #('hello', 'world')
        print(m.expand(r'\3 \2 \1')) #! world hello


    if 0:
        print("search===========")
        # re.search(pattern, string, flags=0)
        # re.search 扫描整个字符串并返回第一个成功的匹配。
        a = "hello world!"
        print(re.search("wor", a)) # <re.Match object; span=(6, 9), match='wor'>


if __name__ == '__main__':
    正则表达式()







