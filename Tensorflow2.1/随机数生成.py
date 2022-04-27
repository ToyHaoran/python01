import tensorflow as tf
import numpy as np

# 返回一个[0,1)之间的随机数
rdm = np.random.RandomState(seed=1)
print(rdm.rand())
print(rdm.rand(2, 3))  # 2行3列的随机数矩阵

# 生成正态分布的随机数，默认均值为0，标准差为1 (维度，mean=均值，stddev=标准差)
d = np.random.randn(2, 2)  # 返回2*2的标准正态随机数
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print("d:", d)

# 生成截断式正态分布的随机数，取值在(μ-2σ， μ+2σ)之内 (维度，mean=均值，stddev=标准差)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print("e:", e)

# 生成均匀分布随机数 (维度，minval=最小值，maxval=最大值)
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print("f:", f)

