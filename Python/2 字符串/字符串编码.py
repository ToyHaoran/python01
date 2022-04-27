if __name__ == '__main__':
    # 字符串 转义字符 (r默认不转义)
    s1, s2, s3, s4 = "abc", "I'm OK", "I\'m \"OK\"", r"I'm \\ OK"
    # 三引号，常用语引用HTML代码，或SQL代码
    helloword = ("""\
        hello
        world""")
    print(helloword)

    # 字符串编码 Python3所有的字符串都是Unicode字符串(1~6字节)。
    print(ord("A"))  # 65 获取字符的整数表示
    print(chr(65))  # A 把编码转换为对应的字符
    print('\u4e2d\u6587')  # 中文

    # 如果要在网络上传输，或者保存到磁盘上，就需要把str(Unicode)变为以字节为单位的bytes
    x = b'abc'  # 每个字符都只占用一个字节
    print('ABC'.encode('ascii'))  # 可以编码为指定的bytes
    print(b'ABC'.decode('ascii'))
    print('中文'.encode('utf-8'))  # 长度为6
    print(b'\xe4\xb8\xad\xe6\x96\x87'.decode('utf-8'))  # 长度为2
    print(b'\xe4\xb8\xad\xe6\xff\x87'.decode('utf-8', errors='ignore'))  # 忽略错误字节

    # Python 没有单独的字符类型，一个字符就是长度为1的字符串。
