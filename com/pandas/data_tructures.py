#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Series创建 = 0
# https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dsintro
# https://www.tutorialspoint.com/python_pandas/python_pandas_series.htm
# Series是一维标记的数组，能够保存任何数据类型（整数，字符串，浮点数，Python对象等）。
if 0:
    print("创建方式======")
    # pandas.Series( data, index, dtype, copy)

    print("1、创建空的series===")
    print(pd.Series())

    print("2、从ndarray创建======")
    s1 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
    print(s1)
    s2 = pd.Series(np.random.randn(5))  # 如果没有传递索引，将创建一个具有值的索引：从0到len(数据)-1
    print(s2)
    s3 = pd.Series(np.array(['a', 'b', 'c', 'd', 'e']))
    print(s3)

    print("3、从dicts创建=====")
    d = {'b': 1, 'a': 0, 'c': 2}
    # 未插入索引，默认将按dict的插入顺序排序（python3.6，pandas0.23以上）
    print(pd.Series(d))
    # 插入索引，将一一对应（NaN(不是数字)是pandas中使用的标准缺失数据标记）
    print(pd.Series(d, index=['b', 'c', 'd', 'a']))

    print("4、从标量值创建=======")
    # 如果data是标量值，则必须提供索引。将重复该值以匹配索引的长度
    print(pd.Series(5., index=['a', 'b', 'c', 'd', 'e']))


Series访问数据 = 0
if 0:
    print("访问数据的方式=======")
    s = pd.Series(np.arange(5), index=['a', 'b', 'c', 'd', 'e'])
    print(s)

    print("1、通过下标访问=========")
    print(s[3])
    print("前三个元素==")
    print(s[:3])
    print("后三个元素==")
    print(s[-3:])
    print("选择下标1到2的元素")
    print(s[1:3])
    print("根据下标选择==")
    print(s[[4, 3, 1]])

    print("2、通过条件获取========")
    print("大于中位数==")
    print(s[s > s.median()])
    print("大于平均值==")
    print(s[s > s.mean()])

    print("3、通过索引标签获取、修改=======")
    print(s["a"])  # 不存在会报错
    print(s.get("f", np.nan))  # 缺少的标签将返回None或指定的默认值
    s["e"] = 100  # 修改某个值
    print(s)


Series矢量化操作和标签对齐 = 0
if 0:
    s = pd.Series(np.arange(1, 6), index=['a', 'b', 'c', 'd', 'e'])
    print("矢量化操作===")
    print(s + s)
    print(s * 2)
    print(np.exp(s))  # e^s 指数函数

    print("自动对齐===")
    print(s[1:] + s[:-1])

    print("name属性===")
    print(pd.Series(np.random.randn(5), name='something').name)
    print(s.rename("lihaoran").name)


DataFrame创建 = 0
# https://pandas.pydata.org/pandas-docs/stable/dsintro.html#dsintro
# https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm
if 0:
    print("创建DF===================")
    # pandas.DataFrame( data, index, columns, dtype, copy)
    print("1、创建一个空DataFrame====")
    print(pd.DataFrame())

    print("2、从列表中创建======")
    data1 = [1, 2, 3, 4, 5]
    print(pd.DataFrame(data1))

    data2 = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
    print(pd.DataFrame(data2))
    print(pd.DataFrame(data2, columns=['Name', 'Age']))
    print(pd.DataFrame(data2, columns=['Name', 'Age'], index=['a', 'b', 'c']))
    print(pd.DataFrame(data2, columns=['Name', 'Age'], index=['a', 'b', 'c'], dtype=float))

    print("3、从字典中创建===========")
    d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
    print(pd.DataFrame(d))  # key是column
    print(pd.DataFrame(d, index=['d', 'b', 'a']))
    print(pd.DataFrame(d, index=['d', 'b', 'a'], columns=['one', 'two', 'three']))

    print("4、通过传递带有日期时间索引和标记列的NumPy数组来创建==========")
    dates = pd.date_range('20130101', periods=6)
    print(dates)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df)

DF数据访问_简单行和列 = 0
if 0:
    print("示例数据=======")
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=list('ABCD'))
    print(df)
    print(df.dtypes)

    print("列操作===================")
    print("选择某列====")
    print(df["A"])
    print("添加某列====")
    df['E'] = ['one', 'two', 'three', 'four', 'five', "six"]
    df["F"] = df['A'] + df["B"]
    print(df)
    print("删除某列====")
    del df["E"]  # 删除
    df.pop("F")  # 弹出
    print(df)

    print("行操作=================")
    print(df[1:3])  # 可以理解为下标是1、2的行（推荐，左毕右开原则）；或者理解为第2、3行。
    print(df['20130102':'20130104'])  # 2,3,4行
    print("添加行========")
    df.loc["new_row"] = [n + 5 for n in range(4)]
    print(df)
    print(df.append(pd.Series({"A": 6, "B": 6}), ignore_index=True))  # 不知道为什么一直报错。。。
    print("删除行======")
    df = df.drop("new_row")
    print(df)

DF数据访问_按标签和位置 = 0
if 0:
    print("行和列同时选择(重点)============")
    print()
    print("按标签选择==============")
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

    print("按位置选择=========")
    print()
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

    print("按条件选择========================")
    print("使用单个列的值来选择数据=====")
    print(df[df.A > 0])  # A列大于0的行
    print("从满足布尔条件的DataFrame中选择值===")
    print(df[df > 0])
    print("使用isin()过滤=======")
    print(df[df["D"].isin([13, 15, 19])])


DF的基本属性 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_basic_functionality.htm
if 0:
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=list('ABCD'))
    print(df)
    # 查看数据
    print("查看轴====")
    print(df.axes)  # 两个轴
    print("是否为空======")
    print(df.empty)
    print("查看维数、大小、形状===")
    print(df.ndim)  # 2
    print(df.size)  # 大小
    print(df.shape)
    print("查看索引、列、基础NumPy数据、数据类型=====")
    print(df.index)
    print(df.columns)
    print(df.values)
    print(df.dtypes)
    print("查看df的顶行和底行====")
    print(df.head(2))  # 前两行数据
    print(df.tail(2))  # 后两行
    print("转置数据======")
    print(df.T)

    print("按轴、值排序========")
    print(df.sort_index(axis=1, ascending=False))
    print(df.sort_values(by='B'))


DF描述性统计 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_descriptive_statistics.htm
if 0:
    d = {'Name': pd.Series(['Tom', 'James', 'Ricky', 'Vin', 'Steve', 'Smith', 'Jack',
                            'Lee', 'David', 'Gasper', 'Betina', 'Andres']),
         'Age': pd.Series([25, 26, 25, 23, 30, 29, 23, 34, 40, 30, 51, 46]),
         'Rating': pd.Series([4.23, 3.24, 3.98, 2.56, 3.20, 4.6, 3.8, 3.78, 2.98, 4.80, 4.10, 3.65])}
    df = pd.DataFrame(d)
    print(df)


    print("求和=====")
    print(df.sum(0))  # 默认0轴
    print(df.sum(1))
    print("平均值====")
    print(df.mean())
    print(df.mean(1))
    print("标准偏差===")
    print(df.std())
    print("常用函数========")
    # S.No.	Function	Description
    # 1 	count()	    Number of non-null observations
    # 2 	sum()	    Sum of values
    # 3 	mean()	    Mean of Values
    # 4 	median()	Median of Values
    # 5 	mode()	    Mode of values
    # 6 	std()	    Standard Deviation of the Values
    # 7 	min()	    Minimum Value
    # 8 	max()	    Maximum Value
    # 9 	abs()	    Absolute Value
    # 10	prod()	    Product of Values
    # 11	cumsum()	Cumulative Sum
    # 12	cumprod()	Cumulative Product
    print("快速统计摘要=======")
    print(df.describe())
    print(df.describe(include="all"))


对元素应用函数 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_function_application.htm
if 1:
    # 要将您自己或其他库的函数应用于Pandas对象，您应该了解三个重要方法
    print("Table-wise Function=========")
    # 对每个元素应用函数
    def adder(ele1, ele2):
        return ele1 + ele2
    df = pd.DataFrame(np.arange(1, 16).reshape(5, 3), columns=['col1', 'col2', 'col3'])
    df = df.pipe(adder, 2)
    print(df)

    print("Row or Column Wise Function============")
    # 沿DataFrame或Panel的轴应用任意函数
    print(df.apply(np.mean, axis=0))
    print(df.apply(np.mean, axis=1))
    print(df.apply(lambda x: x.max() - x.min()))

    print("Element Wise Function===========")
    print(df['col1'].map(lambda x: x * 100))
    print(df.applymap(lambda x: x * 100))


