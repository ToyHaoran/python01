import tensorflow as tf
import numpy as np
"""
传入特征与标签：切分传入张量的第一维度，生成输入特征标签对，构建数据集 (适用Numpy和Tensor)
"""
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))  # (输入特征，标签)
for element in dataset:
    print(element)

"""
函数对指定参数求导，如y=x^2求导
tf.Variable(初始值) 将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))
"""
with tf.GradientTape() as tape:  # with结构记录计算过程
    w = tf.Variable(tf.constant(tf.cast(range(4), tf.float32)))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)  # 求出张量的梯度=2w (函数，对谁求导)
print(grad)  # tf.Tensor([0. 2. 4. 6.], shape=(4,), dtype=float32)

"""
独热编码(one-hot encoding) ：在分类问题中，常用独热码做标签，标记类别: 1表示是，0表示非。
0狗尾草鸢尾 1杂色鸢尾 2弗吉尼亚鸢尾；
标签1表示为[0. 1. 0.] 0%可能是0狗尾草鸢尾；100%可能是1杂色鸢尾；0%可能是2弗吉尼亚鸢尾
"""
labels = tf.constant([1, 0, 2, 1])  # 输入的元素值最小为0，最大为2，共3分类
output = tf.one_hot(labels, depth=3)  # (待转换数据，depth=几分类)
print("result of labels1:", output)

"""
softmax() 使n分类的n个输出(y0,y1,...yn-1)通过softmax()函数符合概率分布；
"""
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print("After softmax, y_pro is:", y_pro)  # y_pro 符合概率分布
print("The sum of y_pro:", tf.reduce_sum(y_pro))  # 通过softmax后，所有概率加起来和为1

"""
参数自更新：赋值操作，更新参数的值并返回。
调用assign_sub前，先用tf.Variable定义变量w为可训练(可自更新)。
"""
w = tf.Variable(4)
w.assign_sub(1)  # (w要自减的内容) 即w=w-1
print("x:", w)  # 4-1=3

