import tensorflow as tf
from keras import models, layers, Model
from keras.layers import Dense
from sklearn import datasets
import numpy as np

"""
六步法搭建神经网络
第一步：import相关模块，如import tensorflow as tf。
第二步：指定输入网络的训练集和测试集，如指定训练集的输入x_train和标签y_train，测试集的输入x_test和标签y_test。
第三步：逐层搭建网络结构，model = models.Sequential()。
第四步：在model.compile()中配置训练方法，选择训练时使用的优化器、损失函数和最终评价指标。
第五步：在model.fit()中执行训练过程，告知训练集和测试集的输入值和标签、每个batch的大小和数据集的迭代次数epoch。
第六步：使用model.summary()打印网络结构，统计参数数目。
"""

# 准备数据
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

"""
用Sequential可以搭建出上层输出就是下层输入的顺序网络结构,
但是无法写出一些带有跳连的非顺序网络结构。这个时候我们可以选择用类class搭建神经网络结构。
"""
class IrisModel(Model):
    # 继承Model
    def __init__(self):
        super(IrisModel, self).__init__()
        # 定义网络结构块
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.d1(x)  # 调用网络结构块，实现前向传播
        return y

model = IrisModel()  # 就是把原来的层封装到了类中，与下面等价
"""
Sequential([网络结构]) 容器，封装了一个神经网络结构。
拉直层：layers.Flatten() 这一层不含计算，只是形状转换，把输入特征拉直变成一维数组
全连接层：layers.Dense(神经元个数，activation= "激活函数“，kernel_regularizer=哪种正则化)
    activation(字符串)可选: relu、softmax、sigmoid、tanh
    kernel_regularizer可选: regularizers.l1()、 regularizers.12()
卷积层：layers.Conv2D() 详见卷积神经网络
LSTM层；layers.LSTM()
"""
# model = models.Sequential()
# model.add(layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2()))

"""
compile配置神经网络的训练方法，告知训练时选择的优化器、损失函数和评测指标
model.compile(optimizer = 优化器, loss = 损失函数, metrics = ["准确率"] )
Optimizer(优化器)可选:
    'sgd' or optimizers.SGD (lr=学习率,momentum=动量参数)
    'adagrad' or optimizers.Adagrad (lr=学习率)
    'adadelta' or optimizers.Adadelta (lr=学习率)
    'adam' or optimizers.Adam (lr=学习率，beta_ 1=0.9, beta_ 2=0.999)
loss是(损失函数)可选:
    'mse' or losses.MeanSquaredError()
    'sparse_categorical_crossentropy' or losses.SparseCategoricalCrossentropy(from_logits=False)
    from_logits参数：是否是原始输出，即没有经概率分布的输出。
        有些神经网络的输出是经过了softmax等函数的概率分布，有些则不经概率分布直接输出，
Metrics(评测指标)可选:
    'accuracy' : y_和y都是数值，如y_=[1] y=[1]
    'categorical_accuracy' : y_和y都是独热码(概率分布)，如y_ =[0,1,0] y=[0 256.0.695,0.048]
    'sparse_categorical_accuracy' : y_是数值，y是独热码(概率分布)，如y_ =[1] y=[0 256,0.695,0.048]
"""
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

"""
model.fit(训练集的输入特征，训练集的标签，
        batch_size=每次喂入神经网络的样本数，推荐个数为：2^n
        epochs=要迭代多少次数据集
        validation_data=(测试集的输入特征，测试集的标签),
        validation_split=从训练集划分多少比例给测试集， 和_data二选一
        validation_freq =多少次epoch使用测试集验证一次结果)
"""
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()  # 打印出网络的结构和参数统计
