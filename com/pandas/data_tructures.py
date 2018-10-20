#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def 对象创建():
    if 1:
        print("创建对象========")
        print("创建一个Series通过传递值的列表============")
        s = pd.Series([1,3,5,np.nan,6,8])
        print(s)

        print("DataFrame通过传递带有日期时间索引和标记列的NumPy数组来创建==========")
        dates = pd.date_range('20130101', periods=6)
        print(dates)
        df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
        print(df)

        print("DataFrame通过传递可以转换为类似系列的对象的dict来创建===========")
        df2 = pd.DataFrame({'A': 1.,
                            'B': pd.Timestamp('20130102'),
                            'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                            'D': np.array([3] * 4, dtype='int32'),
                            'E': pd.Categorical(["test", "train", "test", "train"]),
                            'F': 'foo'})
        print(df2)
        print(df2.dtypes)  # 每个列都是不同的类型
        print()
        print()
        print()

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

    if 1:
        # 选择
        # Getting
        print("选择某列====")
        print(df["A"])
        print("选择某几行：切片======")
        print(df[1:3])  # 可以理解为下标是1、2的行（推荐，左毕右开原则）；或者理解为第2、3行。
        print(df['20130102':'20130104'])  # 2,3,4行
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































if __name__ == '__main__':
    对象创建()




