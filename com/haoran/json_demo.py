#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# json.dumps(): 对数据进行编码。
# json.loads(): 对数据进行解码。

# Python 编码为 JSON 类型转换对应表
# Python	                                JSON
# dict	                                    object
# list, tuple	                            array
# str	                                    string
# int, float, int- & float-derived Enums   	number
# True	                                    true
# False	                                    false
# None	                                    null

# JSON 解码为 Python 类型转换对应表：
# JSON	        Python
# object	    dict
# array	        list
# string	    str
# number (int)	int
# number (real)	float
# true	        True
# false	        False
# null	        None


# Python 字典类型转换为 JSON 对象
data = {
    'no': 1,
    'name': 'Runoob',
    'url': 'http://www.runoob.com'
}

json_str = json.dumps(data)
print("Python 原始数据：", type(data), repr(data))
print("JSON 对象：",type(json_str), json_str)

# 将 JSON 对象转换为 Python 字典
data2 = json.loads(json_str)
print(type(data2))
print ("data2['name']: ", data2['name'])
print ("data2['url']: ", data2['url'])

if 0:
    # 写入 JSON 数据
    with open('data.json', 'w') as f:
        json.dump(data, f)

    # 读取数据
    with open('data.json', 'r') as f:
        data = json.load(f)
