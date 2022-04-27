#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

IO工具_CVS读取 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_io_tool.htm
if 1:
    print("读取csv文件====")
    df = pd.read_csv("temp.csv")
    print(df)
    print("相当于select name,age from df limit 3=====")
    print(df[["Name", "Age"]].head(3))
    print("where条件======")
    print(df[df["Age"] >= 30])

    print("自定义索引========")
    print(pd.read_csv("temp.csv", index_col=['S.No']))

    print("转换类型====")
    print(pd.read_csv("temp.csv", dtype={'Salary': np.float64}))

    print("指定column的名称=====")
    # 相当于数据中没有表头
    print(pd.read_csv("temp.csv", names=['a', 'b', 'c', 'd', 'e']))
    print(pd.read_csv("temp.csv", names=['a', 'b', 'c', 'd', 'e'], header=0))

    print("跳过指定的行数=====")
    print(pd.read_csv("temp.csv", skiprows=2))

IO工具_CVS写入 = 0
if 1:
    pass