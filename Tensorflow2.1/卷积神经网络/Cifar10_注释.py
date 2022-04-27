import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, \
    Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model

"""
cifar10数据集一共有6万张彩色图片，每张图片有32行32列像素点的红绿蓝三通道数据。
提供5万张32*32像素点的十分类彩色图片和标签，用于训练。
提供1万张32*32像素点的十分类彩色图片和标签，用于测试。
十个分类分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车，分别对应标签0、1、2、3一直到9
"""

np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        # 一层卷积层
        """
        tf描述卷积层 layers.Conv2D (
            filters =卷积核个数，
            kernel_size =卷积核尺寸(核高h,核宽w)， 或正方形写核长整数 
            strides =滑动步长(纵向步长h，横向步长w)，或横纵向相同写步长整数，默认1
            padding = "same" or "valid", # 使用全零填充是"same",不使用是"valid" (默认)
            activation ="relu" or "sigmoid" or "tanh" or "softmax"等，# 如有BN此处不写
            input_shape = (高，宽，通道数)  # 输入特征图维度，可省略
            )
        """
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        """ 池化层
        tf.keras.layers.MaxPool2D, AveragePooling2D(
            pool_size=池化核尺寸, # 正方形写核长整数，或(核高h，核宽w)
            strides=池化步长, #步长整数，或(纵向步长h,横向步长w)，默认为pool_size
            padding='valid'or'same' #使用全零填充是"same"，不使用是"valid" (默认)
        )
        """
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层
        # 两层 全连接层
        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5),
                         activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y
class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y

class ConvBNRelu(Model):
    """
     卷积连接器把这四个分支按照深度方向堆叠在一起，构成Inception结构块的输出，
     由于Inception结构块中的卷积操作均采用了CBA结构，即先卷积再BN再采用relu激活函数，
     所以将其定义成一个新的类ConvBNRelu，减少代码长度，增加可读性。
    """
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        """
        :param ch: 代表特征图的通道数，也即卷积核个数;
        :param kernelsz: 代表卷积核尺寸;
        :param strides: 代表卷积步长;
        :param padding: 代表是否进行全零填充。
        """
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False) #在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
        return x
class InceptionBlk(Model):
    """
    完成ConvBNRelu之后，就可以开始构建 InceptionNet 的基本单元了，
    同样利用 class 定义的方式，定义一个新的 InceptionBlk类。
     InceptionNet 网络的主体就是由其基本单元构成的，有了Inception结构块后，
     就可以搭建出一个精简版本的InceptionNet，网络共有10层，其模型结构如图
    """
    def __init__(self, ch, strides=1):
        """
        与 ConvBNRelu 类中一致
        :param ch:  仍代表通道数
        :param strides: 代表卷积步长
        """
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        """
        concat along axis=channel
        函数将四个输出按照深度方向连接在一起，x1、x2_2、x3_2、x4_2 分别代表四列输出，
        结合结构图和代码很容易看出二者的对应关系。
        """
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x
class Inception10(Model):
    """
    InceptionNet 网络的主体就是由其基本单元构成的，有了Inception结构块后，就可以搭建出一个精简版本的InceptionNet，
    网络共有10层，其模型结构如图
    第一层采用16个3*3卷积核，步长为1，全零填充，BN操作，rule激活。
    随后是4个Inception结构块顺序相连，每两个Inception结构块组成block，
        每个block中的第一个Inception结构块，卷积步长是2，第二个Inception结构块，卷积步长是1，
        这使得第一个Inception结构块输出特征图尺寸减半，因此把输出特征图深度加深，尽可能保证特征抽取中信息的承载量一致。
    block_0设置的通道数是16，经过了四个分支，输出的深度为4*16=64；
    在self.out_channels *= 2给通道数加倍了，所以block_1通道数是block_0通道数的两倍是32，
    经过了四个分支，输出的深度为4*32=128，这128个通道的数据会被送入平均池化，送入10个分类的全连接。
    """
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        """
        InceptionNet 网络不再像 VGGNet 一样有三层全连接层(全连接层的参数量占 VGGNet 总参数量的 90 %)，
        而是采用“全局平均池化+全连接层”的方式，这减少了大量的参数。
        :param num_blocks: 代表 InceptionNet 的 Block 数，每个 Block 由两个基本单元构成；
        :param num_classes: 代表分类数，对于 cifar10 数据集来说即为 10;
        :param init_ch: 代表初始通道数，也即 InceptionNet 基本单元的初始卷积核个数。
        :param kwargs:
        """
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch

        # 第一层采用16个3*3卷积核，步长为1，全零填充，BN操作，rule激活。
        # 设定了默认init_ch=16,默认输出深度是16，
        # 定义ConvBNRe lu类的时候，默认卷积核边长是3步长为1全零填充，所以直接调用
        self.c1 = ConvBNRelu(init_ch)

        # 每个bIock中的第一个Inception结构块，卷积步长是2，
        # 第二个Inception结构块，卷积步长是1，
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block

            # 给通道数加倍了，所以block_1通道数是block_0通道数的两倍是32
            self.out_channels *= 2

        self.p1 = GlobalAveragePooling2D()  # 128个通道的数据会被送入平均池化
        self.f1 = Dense(num_classes, activation='softmax')  # 送入10个分类的全连接。

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out
class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y

model = Baseline()  # 基础CNN模型
# model = LeNet5()
# model = Inception10(num_blocks=2, num_classes=10)  # Block数是2，block_0和block_1; 网络10分类
# model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "Baseline.ckpt"
# checkpoint_save_path = "./checkpoint/LeNet5.ckpt"  # 这里不同模型需要修改一下
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5,
                    validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

# print(model.trainable_variables)
"""
baseline/conv2d/kernel:0 (5, 5, 3, 6)记录了第层网络用的5*5*3的卷积核，一共6个，下边给出了这6个卷积核中的所有参数W；
baseline/conv2d/bias:0 (6,)这里记录了6个卷积核各自的偏置项b，每个卷积核一个 b，6个卷积核共有6个偏置6 ；
baseline/batch_normalization/gamma:0 (6,)，这里记录了BN操作中的缩放因子γ，每个卷积核一个γ，一个6个γ；
baseline/batch_normalization/beta:0 (6,)，里记录了BN操作中的偏移因子β，每个卷积核一个β，一个6个β；
baseline/dense/kernel:0 (1536, 128)，这里记录了第一层全链接网络，1536 行、128列的线上的权量w；
baseline/dense/bias:0 (128,)，这里记录了第一层全连接网络128个偏置b；
baseline/dense_1/kernel:0 (128, 10)，这里记录了第二层全链接网络，128行、10列的线上的权量w；
baseline/dense_1/bias:0 (10,)，这里记录了第二层全连接网络10个偏置b。
"""
file = open('weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
