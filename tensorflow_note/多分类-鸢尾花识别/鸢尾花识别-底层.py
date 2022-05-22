import time
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

"""
利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线
(有点底层源码的感觉，重点记忆，背下来)

数据集介绍：共有数据150组，每组包括花尊长、花尊宽、花瓣长、花瓣宽4个输入特征。以及这组特征对应的鸢尾花类别。
类别包括(狗尾草鸢尾)(杂色鸢尾)(弗吉尼亚鸢尾)三类，分别用数字0，1，2表示。

1 准备数据
    数据集读入
    数据集乱序
    生成训练集和测试集
    配成(输入特征，标签)对，每次读入一小撮(batch)
2 搭建网络
    定义神经网路中所有可训练参数
3 参数优化
    嵌套循环迭代，with结构更新参数，显示当前loss
4 测试效果
    计算当前参数前向传播后的准确率，显示当前acc
5 acc / loss可视化
"""

# 导入数据，分别为输入特征和标签
data = datasets.load_iris().data  # .data返回iris数据集所有输入特征
label = datasets.load_iris().target  # .target返回iris数据集所有标签

# 随机打乱数据(因为原始数据是顺序的，顺序不打乱会影响准确率)
# seed: 随机数种子，可以为任何整数，当设置之后，每次生成的随机数都一样，都是按116号方式随机的；
# 如果不指定，每次训练结果都不一样；如果三个地方不相同，输入特征和标签就对应不上了；
np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(label)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = data[:-30]
y_train = label[:-30]
x_test = data[-30:]
y_test = label[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# 使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 定义神经网络所有可训练参数(用Variable标记)
# 4个输入特征，故输入层为4个输入节点；因为3分类，故输出层为3个神经元
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))  # 权重4行3列的张量
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))  # 偏置

# 超参数：不同于权重w和偏置b，需要人工设定；
lr = 0.1  # 学习率为0.1 (可使用指数衰减学习率)
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环多少轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# 训练部分
now_time = time.time()  # 记录开始时间
for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息
            # 前向传播过程 计算y
            y = tf.matmul(x_train, w1) + b1  # y为预测结果
            y = tf.nn.softmax(y)  # 使输出y符合概率分布(此操作后与独热码同量级，可相减求loss)
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            # loss_regularization = tf.reduce_sum([tf.nn.l2_loss(w1)])  # 添加l2正则化防止过拟合
            # loss = loss + 0.03 * loss_regularization  # REGULARIZER = 0.03
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确

        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        # 这里可以修改为其他优化器如SGDM、AdaGrad等(tensorflow已封装)
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1  # 使用更新后的参数进行预测
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        pred = tf.cast(pred, dtype=y_test.dtype)  # 将pred转换为test_label的数据类型
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)  # 将每个batch的correct数加起来
        total_correct += int(correct)  # 将所有batch中的correct数加起来
        # total_number为测试的总样本数，也就是test_data的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]

    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")
total_time = time.time() - now_time  # 记录结束时间
print("total_time", total_time)  # 打印训练时间

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出train_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()
