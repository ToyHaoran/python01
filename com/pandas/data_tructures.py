#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Series创建方式 = 0
# 参考：https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dsintro
# Series是一维标记的数组，能够保存任何数据类型（整数，字符串，浮点数，Python对象等）。
# 轴标签统称为索引。
if 0:
    print("创建方式======")

    print("1、来自ndarray======")
    s1 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    s2 = pd.Series(np.random.randn(5)) # 如果没有传递索引，将创建一个具有值的索引
    print(s1, s2)

    print("2、从dicts实例化=====")
    d = {'b': 1, 'a': 0, 'c': 2}
    # 未插入索引，默认将按dict的插入顺序排序（python3.6，pandas0.23以上）
    print(pd.Series(d))
    # 插入索引，将一一对应（NaN（不是数字）是pandas中使用的标准缺失数据标记）
    print(pd.Series(d, index=['b', 'c', 'd', 'a']))

    print("3、从标量值(scalar value)=======")
    # 如果data是标量值，则必须提供索引。将重复该值以匹配索引的长度
    print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))

访问数据的方式 = 0
if 0:
    print("访问数据的方式=======")
    s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print(s)
    print("通过下标访问====")
    print(s[0])
    print(s[:3])
    print(s[1:3])
    print(s[[4,3,1]])
    print("大于中位数")
    print(s[s > s.median()])
    print("大于平均值")
    print(s[s > s.mean()])

    print("通过索引标签获取、修改====")
    print(s["a"]) # 不存在会报错
    print(s.get("f", np.nan)) # 缺少的标签将返回None或指定的默认值
    s["e"] = 12 # 修改某个值
    print(s)

矢量化操作和标签对齐 = 0
if 0:
    s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print("矢量化操作===")
    print(s + s)
    print(s * 2)
    print(np.exp(s))

    print("自动对齐===")
    print(s[1:] + s[:-1])

    print("name属性===")
    print(pd.Series(np.random.randn(5), name='something').name)
    print(s.rename("lihaoran").name)

DataFrame创建方式 = 0
# 参考：https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dsintro
# 与Series类似，DataFrame接受许多不同类型的输入
# pandas.DataFrame( data, index, columns, dtype, copy)
if 1:
    print("DataFrame通过传递带有日期时间索引和标记列的NumPy数组来创建==========")
    dates = pd.date_range('20130101', periods=6)
    print(dates)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
    print(df)

    print("从列表中创建======")
    data1 = [2, 2, 2, 2, 2]
    print(pd.DataFrame(data1))
    data2 = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
    print(pd.DataFrame(data2))
    print(pd.DataFrame(data2,columns=['Name','Age']))
    print(pd.DataFrame(data2,columns=['Name','Age'],index=['a', 'b', 'c']))

    print("从字典dict中创建===========")
    d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
    print(pd.DataFrame(d)) # key是column
    print(pd.DataFrame(d, index=['d', 'b', 'a']))
    print(pd.DataFrame(d, index=['d', 'b', 'a'], columns=['one', 'two', 'three']))

    df2 = pd.DataFrame({'A': 1.,
                        'B': pd.Timestamp('20130102'),
                        'C': pd.Series([1, 2, 3, 4], index=list(range(4)), dtype='float32'),
                        'D': np.array([3] * 4, dtype='int32'),
                        'E': pd.Categorical(["test", "train", "test", "train"]),
                        'F': 'foo'})
    print(df2)
    print(df2.dtypes)  # 每个列都是不同的类型





if 0:
    # 查看数据
    print("查看df的顶行和底行====")
    print(df.head(2)) # 前两行数据
    print(df.tail(2)) # 后两行
    print("显示索引，列和基础NumPy数据=====")
    print(df.index)
    print(df.columns)
    print(df.values)
    print("快速统计摘要=======")
    print(df.describe())
    print("转置数据======")
    print(df.T)
    print("按轴、值排序========")
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by='B'))
    print()
    print()
    print()

选择数据 = 0
if 0:
    # 选择
    # Getting
    print("选择某列====")
    print(df["A"])
    print("选择某几行：切片======")
    print(df[1:3])  # 可以理解为下标是1、2的行（推荐，左毕右开原则）；或者理解为第2、3行。
    print(df['20130102':'20130104'])  # 2,3,4行
    print()
    print()
    print()
    # 按标签选择
    print("使用标签获取横截面======")
    print(df.loc[dates[0]])  # 注意是中括号
    print("使用标签选择多轴=========")
    print(df.loc[dates[1], ['A', 'B']])
    print(df.loc['20130102', ['A', 'B']])
    print(df.loc[:, ['A', 'B']])
    print(df.loc['20130102':'20130104', ['A', 'B']])  # 2、3、4行的A、B列
    print("获取标量值=======")
    print(df.loc[dates[0], 'A'])  # 某个具体的值
    print(df.at[dates[0], 'A'])  # 同上
    print()
    print()
    print()
    # 按位置选择
    print("通过传递的整数的位置选择====")
    print(df.iloc[3])  # 下标3
    print("通过整数切片====")
    print(df.iloc[3:5, 0:2])  # 行下标3、4；列下标0、1
    print("通过整数位置列表======")
    print(df.iloc[[1, 2, 4], [0, 2]])
    print("明确切片行=====")
    print(df.iloc[1:3, :])
    print("明确切片列===")
    print(df.iloc[:, 1:3])
    print("明确获取值===")
    print(df.iloc[1, 1])
    print(df.iat[1, 1])
    print()
    print()
    # 布尔索引
    print("使用单个列的值来选择数据=====")
    print(df[df.A > 0])  # A列大于0的行
    print("从满足布尔条件的DataFrame中选择值===")
    print(df[df > 0])
    print("使用isin()过滤=======")
    df2 = df.copy()
    df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
    print(df2)
    print(df2[df2["E"].isin(["two", "four"])])
