#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建一个Series通过传递值的列表
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

# DataFrame通过传递带有日期时间索引和标记列的NumPy数组来创
dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)

# DataFrame通过传递可以转换为类似系列的对象的dict来创建
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
print(df2)




