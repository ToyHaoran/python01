#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DF数据访问_简单行和列 = 0
if 0:
    print("示例数据=======")
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape(6, 4), index=dates, columns=list('ABCD'))

    print("列操作===================")
    print("选择某列====")
    print("注意：只要选择一列就是一个Series========")
    print(df["A"])
    print(type(df["A"]))  # <class 'pandas.core.series.Series'>
    print(type(df.people))  # 同上
    print(df[["A", "B"]])
    print(type(df[["A", "B"]]))  # <class 'pandas.core.frame.DataFrame'>
    print("添加某列====")
    df['E'] = ['one', 'two', 'three', 'four', 'five', "six"]
    df["F"] = df['A'] + df["B"]
    df.insert(0, "第一列", ['one', 'two', 'three', 'four', 'five', "six"])

    print(df)
    print("删除某列====")
    del df["E"]  # 删除
    df.pop("F")  # 弹出
    df = df.drop("第一列", axis=1)  # 通过轴删除行或者列
    print(df)

    print("行操作=================")
    print(type(df[:1])) # <class 'pandas.core.frame.DataFrame'>
    print(df[1:3])  # 可以理解为下标是1、2的行（推荐，左毕右开原则）；或者理解为第2、3行。
    print(df['20130102':'20130104'])  # 2,3,4行
    print("添加行========")
    df.loc["new_row"] = [n + 5 for n in range(4)]
    print(df)
    print(df.append(pd.Series({"A": 6, "B": 6}), ignore_index=True))  # 不知道为什么一直报错。。。
    print("删除行======")
    df = df.drop("new_row", axis=0)  # 默认0
    print(df)

DF数据访问_按标签 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_indexing_and_selecting_data.htm
if 0:
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df)
    print("按标签选择==============")
    print("选择某行======")
    print(df.loc[dates[0]])  # 注意是中括号

    print("选择多轴=========")
    print(df.loc[dates[1], ['A', 'B']])
    print(df.loc['20130102', ['A', 'B']])
    print(df.loc[:, ['A', 'B']])
    # 注意加[]表示多列，返回的是DF
    print(df.loc[:, ['A']])
    # 不加，返回Series，可以用unique()方法
    print(df.loc[:, 'A'].unique())
    print(df.loc['20130102':'20130104', ['A', 'B']])  # 2、3、4行的A、B列

    print("选择某个具体值=======")
    print(df.loc[dates[0], 'A'])  # 某个具体的值
    print(df.at[dates[0], 'A'])  # 同上

    print("按条件选择=======")
    print(df.loc["20130102"] > 0)
    print(df.loc[:, "A"] > 0)
    print(df[df.loc[:, "A"] > 0])  # A列大于0的行
    print(df[df.people > 0])   # 同上
    # 如何设置多个过滤条件？？
    # print(df[(df.loc[:, "A"] > 0) and (df.loc[:, "B"] > 0)])

    print("使用isin()过滤=======")
    print(df[df["D"].isin([0.185162, 0.131633, 0.818331])])


DF数据访问_按位置 = 0
if 0:
    dates = pd.date_range('20130101', periods=6)
    df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
    print(df)
    print("按位置选择=========")
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

    # ix方法已经被遗弃了，不使用
    # 用reindex代替


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
if 0:
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

重建索引和轴 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_reindexing.htm
if 0:
    print("reindex==========")
    N = 20
    df = pd.DataFrame({
        'A': pd.date_range(start='2016-01-01', periods=N, freq='D'),
        'x': np.linspace(0, stop=N - 1, num=N),
        'y': np.random.rand(N),
        'C': np.random.choice(['Low', 'Medium', 'High'], N).tolist(),
        'D': np.random.normal(100, 10, size=N).tolist()
    })
    print(df.reindex(index=[1, 2, 3], columns=['A', 'C', 'B']))

    df1 = pd.DataFrame(np.random.randn(4, 3), columns=['col1', 'col2', 'col3'])
    df2 = pd.DataFrame(np.random.randn(2, 3), columns=['col1', 'col2', 'col3'])
    print(df2.reindex_like(df1))
    # ffill填充前面的值 # backfill填充后面的值. nearest从最近的索引值填充
    print(df2.reindex_like(df1, method='ffill', limit=1))

    print("重命名index或axis========")
    print(df1)
    print(df1.rename(columns={'col1': 'c1', 'col2': 'c2'}, index={0: 'apple', 1: 'banana', 2: 'durian'}))


迭代 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_iteration.htm
if 0:
    print("迭代========")
    df = pd.DataFrame(np.random.randn(4, 3), columns=['col1', 'col2', 'col3'])

    print("迭代列名==========")
    for col in df:
        print(col)

    print("iteritems迭代列=========")
    for key, value in df.iteritems():
        print(key, value)  # value为Series对象

    print("iterrows迭代行==========")
    for row_index, row in df.iterrows():
        print(row_index, row)

    print("itertuples迭代行=========")
    # 返回一个迭代器，为DataFrame中的每一行产生一个命名元组。元组的第一个元素是行的相应索引值，而其余值是行值
    for row in df.itertuples():
        print(row)
    # 迭代时不要尝试修改任何对象。迭代用于读取，迭代器返回原始对象（视图）的副本，因此更改不会反映在原始对象上

排序 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_sorting.htm
if 0:
    print("排序======")
    # 按lable排序，或者按实际值排序
    unsorted_df = pd.DataFrame(np.random.randn(10, 2), index=[1, 4, 6, 2, 3, 5, 9, 8, 0, 7], columns=['col2', 'col1'])
    print(unsorted_df)

    print("按index排序==========")
    print(unsorted_df.sort_index(ascending=False))  # 默认按0轴排序
    print(unsorted_df.sort_index(axis=1))
    print(unsorted_df.sort_index(axis=1).sort_index())

    print("按值排序========")
    unsorted_df = pd.DataFrame({'col1': [2, 1, 1, 1], 'col2': [1, 3, 2, 4]})
    print(unsorted_df.sort_values(by='col1'))
    print(unsorted_df.sort_values(by=['col1', 'col2']))
    # 从mergesort，heapsort和quicksort中选择算法
    print(unsorted_df.sort_values(by=['col1', 'col2'], kind="mergesort"))

字符串操作 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_working_with_text_data.htm
if 0:
    # 将Series Object转换为String Object，然后执行操作
    # 和str的内嵌函数通用
    s = pd.Series(['Tom ', 'William Rick ', 'John', 'Alber@t', np.nan, '1234', 'Steve Smith'])
    print(s.str.lower())
    print(s.str.upper())
    print(s.str.len())
    print(s.str.strip())
    print(s.str.split())
    print(s.str.cat(sep="_"))  # 连接
    print(s.str.contains(' '))
    print(s.str.replace('@', '$'))
    print(s.str.repeat(2))  # 重复两次
    print(s.str.count('m'))  # 每个字符串中出现的次数
    print(s.str.startswith('T'))
    print(s.str.endswith('t'))
    print(s.str.find('e'))
    print(s.str.findall('e'))
    print(s.str.swapcase())

自定义打印区域 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_options_and_customization.htm
if 0:
    df = pd.DataFrame(np.random.randn(70, 20))
    print(df)
    print(pd.get_option("display.max_rows"))
    print(pd.get_option("display.max_columns"))
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.max_columns", 10)
    # print(df)
    print(pd.describe_option("display.max_rows"))
    print(pd.describe_option("display.max_columns"))
    # S.No	参数	                        描述
    # 1	    display.max_rows	        显示要显示的最大行数
    # 2	    display.max_columns	        显示要显示的最大列数
    # 3	    display.expand_frame_repr	将数据框显示为拉伸页面
    # 4	    display.max_colwidth	    显示最大列宽
    # 5	    display.precision	        显示十进制数的精度

静态函数 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_statistical_functions.htm
if 0:
    print("pct_change=======")
    # 将每个元素与其先前元素进行比较，并计算更改百分比
    s = pd.Series([1, 2, 3, 4, 5, 4])
    print(s.pct_change())
    df = pd.DataFrame(np.random.randn(5, 2))
    print(df.pct_change())  # 默认对列进行操作

    print("协方差====")
    s1 = pd.Series(np.random.randn(10))
    s2 = pd.Series(np.random.randn(10))
    print(s1.cov(s2))

    frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    print(frame['a'].cov(frame['b']))
    print(frame.cov())
    # 第一个语句中a和b列之间的cov，同样是DataFrame上cov返回的值

    print("相关性========")
    # 相关性显示任意两个值数组（系列）之间的线性关系。有多种方法可以计算相关性，如pearson（默认值），spearman和kendall
    frame = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    print(frame['a'].corr(frame['b']))
    print(frame.corr())

    print("Data Ranking")
    # 为元素数组中的每个元素生成排名。如果是关系，则指定平均等级????
    s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))
    s['d'] = s['b']
    print(s.rank())

窗口函数 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_window_functions.htm
# https://blog.csdn.net/maymay_/article/details/80241627
# https://blog.csdn.net/xxzhangx/article/details/76938053
if 0:
    print("rolling==========")
    s = pd.Series([1, 2, 3, 5, 6, 10, 12, 14, 12, 30])
    print(s)
    print(s.rolling(window=3).mean())
    # 首先我们设置的窗口window=3，也就是3个数取一个均值。
    # index 0,1 为NaN，是因为它们前面都不够3个数，
    # 等到index2 的时候，就是（index0+index1+index2 ）/3
    # index3 的值就是（index1+index2+index3）/ 3

    df = pd.DataFrame(np.random.randn(10, 4),
                      index=pd.date_range('1/1/2000', periods=10),
                      columns=['A', 'B', 'C', 'D'])
    print(df.rolling(window=3).mean())

    print("expanding==跳过===")
    print(df.expanding(min_periods=3).mean())

    print("ewm==跳过=====")
    print(df.ewm(com=0.5).mean())

聚合 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_aggregations.htm
if 0:
    df = pd.DataFrame(np.arange(40).reshape(10, 4),
                      index=pd.date_range('1/1/2000', periods=10),
                      columns=['A', 'B', 'C', 'D'])
    print(df)
    r = df.rolling(window=3, min_periods=1)
    print("全部元素都进行聚合=====")
    print(r.aggregate(np.sum))  # 每三个数进行求和
    print("单个列上应用聚合====")
    print(r['A'].aggregate(np.sum))
    print("多个列上应用聚合=======")
    print(r[['A', 'B']].aggregate(np.sum))
    print("多个列上应用多个函数====")
    print(r['A', "B"].aggregate([np.sum, np.mean]))
    print("将不同的函数应用于不同列")
    print(r.aggregate({'A': np.sum, 'B': np.mean}))

NaN缺失数据的处理 = 0
# https://www.tutorialspoint.com/python_pandas/python_pandas_missing_data.htm
if 0:
    print("缺失数据的处理=====")
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'], columns=['one', 'two', 'three'])
    df = pd.DataFrame(np.arange(15).reshape(5, 3), index=['a', 'c', 'e', 'f', 'h'], columns=['one', 'two', 'three'])
    df = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    print(df)
    print(df['one'].isnull())
    print(df['one'].notnull())

    print("缺少数据的计算=========")
    # 在对数据求和时，NA将被视为零;如果数据都是NA，那么结果将是NA
    print(df['one'].sum())

    print("用标量值替换NaN==========")
    print(df.fillna(0))  # 用0填充
    print(df.fillna(999))  # 用999填充
    print(df.replace({np.nan: 111, 10.: 1000, 11.: 1100}))
    print(df.fillna(method="pad"))  # 用前面的填充
    print(df.fillna(method="backfill"))  # 用后面的填充

    print("删除缺失值====")
    print(df.dropna())

    print("稀疏对象====")
    # 占用更少的内存
    # https://www.tutorialspoint.com/python_pandas/python_pandas_sparse_data.htm
    ts = pd.Series(np.random.randn(10))
    ts[2:-2] = np.nan
    sts = ts.to_sparse()
    print(sts)
    print("密度：", sts.density)

    # 调用to_dense可以将任何稀疏对象转换回标准密集形式-
    print(sts.to_dense())

