if __name__ == '__main__':
    str1 = "hello"
    str2 = "123"
    print(str1.capitalize())  # 首字母大写 Hello
    print(str1.center(9, "-"))  # 两边扩展字符 --hello--
    print(str1.count("ll", 0, 5))  # 字母出现的次数 1
    print(str1.startswith("he", 0, 10))  # 字符串是不是以he开头 True
    print(str1.endswith("lo", 0, 10))  # 字符串是不是以lo结尾 True
    print(str1.find("e"))  # 1
    print(str1.rfind("e"))  # 1
    print(str1.index("e"))  # 1
    print(str1.rindex("e"))  # 1
    print((str1 + str2).isalnum())  # 所有字符都是字母数字返回True
    print(str2.isdigit())
    print(str1.isalpha())
    print(str1.islower())
    print(str1.isupper())  # 字符串是不是大写
    print(str1.istitle())
    print(len(str1))
    str3 = "-"
    print(str3.join(("aaa", "bbb", "ccc")))
    print(str1.lower())
    print(str1.title())  # 单词首字母大写,其他变为小写
    print(str1.upper())  # 字符串变大写
    print(str1.swapcase())
    print(str1.ljust(10, "-"))
    print(str1.rjust(10, "-"))
    print("12".zfill(5))  # str.zfill()用零填充左侧的数字字符串
    print(str1.lstrip("he"))
    print(str1.rstrip("lo"))
    print(str1.strip("o"))
    print(max(str1))
    print(min(str1))
    print(str1.replace("he", "HE", 1))
    print(str1.split("e"))