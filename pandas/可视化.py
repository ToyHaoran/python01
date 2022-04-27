#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

可视化 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_visualization.htm
if 1:
    print("plot========")
    df = pd.DataFrame(np.random.randn(10, 4), index=pd.date_range('1/1/2000', periods=10), columns=list('ABCD'))
    df.plot()  # 1

    print("条形图======")
    df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
    df.plot.bar()  # 2
    df.plot.bar(stacked=True)  # 3
    df.plot.barh(stacked=True)  # 4

    print("直方图===")
    df = pd.DataFrame({'a': np.random.randn(1000) + 1, 'b': np.random.randn(1000), 'c': np.random.randn(1000) - 1},
                      columns=['a', 'b', 'c'])
    df.plot.hist(bins=20)  # 5

    print("方块图===")
    df = pd.DataFrame(np.random.rand(10, 5), columns=['A', 'B', 'C', 'D', 'E'])
    df.plot.box()  # 6

    print("区域图===")
    df = pd.DataFrame(np.random.rand(10, 4), columns=['a', 'b', 'c', 'd'])
    df.plot.area()  # 7

    print("散点图===")
    df = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
    df.plot.scatter(x='a', y='b')  # 8

    print("饼图===")
    df = pd.DataFrame(3 * np.random.rand(4), index=['a', 'b', 'c', 'd'], columns=['x'])
    df.plot.pie(subplots=True)  # 9

    plt.show()  # 如果不加这句，则必须在Python Console中才能打开