import numpy as np

"""
数学上将一维数组称为向量，将二维数组称为矩阵，将三维及以上的数组称为“张量tensor”或“多维数组”。
"""
# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2], [3, 4]])
c = np.arange(-10, 10, 2)
c.reshape(2, 5)  # 变为二维数组
c.flatten()  # 降为一维数组
print(np.ones(3))  # [1. 1. 1.]
print(np.zeros((2, 3)))  # 二维数组 两行三列 [[0. 0. 0.] [0. 0. 0.]]

# 数组属性(具体见张量介绍一节)
print(a.shape, b.shape)  # (5,) (2, 2)

# 数组切片(类似列表切片)
print(b[1][1])
print(b[:, 1])  # 提取 列下标=1 的所有元素
y = c > 0  # 布尔数组
print(y.astype(np.int))  # [0 0 0 0 0 0 1 1 1 1]
print(c[c > 0])  # 过滤大于0的元素

# 数组运算
a = np.array([[1, 2],
              [3, 4]])
b = np.array([[5, 6],
              [7, 8]])
print(a.shape, a.ndim)  # (2, 2) 2维
print(a + b)
print(a - b)
print(a * b)  # 对应元素乘法 element-wise product
print(a.dot(b), np.dot(a, b), sep="\n")  # 矩阵乘法 Matrix Product
print(a / b)
print(np.vstack((a, b)))  # 两个数组按垂直方向叠加

# 广播 低维c扩展到高维a相同的形状
print(10 * np.sin(a))
c = np.array([10, 20])
print(a * c)

# 生成等间隔数值点
x, y = np.mgrid[1:3:1, 2:4:0.5]  # [起始值:结束值:步长，起始值:结束值:步长，... ]  左闭右开
x, y = x.ravel(), y.ravel()  # 将x, y变成一维数组
grid = np.c_[x, y]  # 合并配对为二维张量，生成二维坐标点 (感觉像zip)
print('grid:\n', grid)
