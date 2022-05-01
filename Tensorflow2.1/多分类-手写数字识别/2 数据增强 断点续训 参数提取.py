import os
import numpy as np
from tensorflow.keras import callbacks, datasets, models, layers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

mnist = datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 给数据增加一个维度,从(60000, 28, 28)reshape为(60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

"""
数据增强：对图像的增强就是对图像进行简单形变，解决因为拍照角度不同等因素造成的影响。
image_gen._train = lmageDataGenerator(
	rescale =所有数据将乘以该数值
	rotation_ range =随机旋转角度数范围
	width_ shift range =随机宽度偏移量
	height shift range =随机高度偏移量
	水平翻转: horizontal_flip =是否随机水平翻转
	随机缩放: zoom_range =随机缩放的范围[1-n, 1+n] )
"""
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)

model = models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

"""
断点续训，存取模型  load_weights(路径文件名)
1 定义存放模型的路径和文件名，命名为ckpt文件
2 生成ckpt文件时会同步生成index索引表，所以判断索引表是否存在，来判断是否存在模型参数
3 如有索引表，则直接读取ckpt文件中的模型参数
"""
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):   
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                        save_weights_only=True,
                                        save_best_only=True)

model.fit(image_gen_train.flow(x_train, y_train, batch_size=32),  # 数据增强
          epochs=5, validation_data=(x_test, y_test), validation_freq=1,
          callbacks=[cp_callback])  # 保存模型
model.summary()

"""
参数提取：把参数存入文本
model.trainable_variables：返回模型中可训练的参数
"""
np.set_printoptions(threshold=np.inf)  # (threshold=超过多少省略显示)  inf表示无限大
print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()
