import tensorflow as tf
import numpy as np

"""
维数	阶	   名字    	例子
0-D	    0	标量 scalar	s=1 2 3
1-D	    1	向量 vector	v=[1, 2, 3]
2-D	    2	矩阵 matrix	m=[[1,2,3],[4,5,6][7,8,9]]
N-D	    N	张量 tensor	t=[[[ 有几个中括号就是几阶张量

张量类型：tf.int32(默认), tf.float32, tf.float64, tf.bool, tf.string,
"""

# 创建Tensor
a = tf.constant([1, 5], dtype=tf.int64)
b = tf.cast(a, dtype=tf.int32)  # 强制转换类型
print(b)

# 将Numpy数组转为Tensor
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print("a:", a)
print("转为张量:", b)
print("转为Numpy：", b.numpy())

"""
填充张量(维度,指定值)  
一维直接写个数  二维用[行，列]  多维用[n,m,.K....]
"""
a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print("a:", a)
print("b:", b)
print("c:", c)

"""
对应元素的四则运算(必须维度相同): tf.add(张量1，张量2), tf.subtract, tf.multiply, tf.divide
平方、次方与开方: tf.square(张量1), tf.pow, tf.sqrt
矩阵乘: tf.matmul(矩阵1, 矩阵2)
"""

"""
理解axis
在一个二维张量或数组中，可以通过调整axis等于0或1控制执行维度。
axis=0代表跨行(经度，down),而axis=1代表跨列(纬度，across)
"""
x = tf.constant([[1, 2, 3], [2, 2, 3]])
print("x:", x)
print("所有数的均值:", tf.reduce_mean(x))  # 不指定axis,则所有元素参与计算
print("每一行的和:", tf.reduce_sum(x, axis=1))
print("每行最小值：", tf.reduce_min(x, axis=1))
print("每列最大值:", tf.reduce_max(x, axis=0))
print("每列最大值的索引：", tf.argmax(x, axis=0))

# where(条件语句，真返回A，假返回B) 类似三元运算符
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
print("c：", c)  # [1 2 3 4 5]
