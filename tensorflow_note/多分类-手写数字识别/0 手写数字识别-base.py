import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers, regularizers, losses, Model

# 加载数据集 train_images.shape=(60000, 28, 28) test_images.shape=(10000, 28, 28)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 正规化：像素在0~255之间，映射到0~1
train_images, test_images = train_images / 255.0, test_images / 255.0

class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y

model = MnistModel()  # 与下面等价
# model = models.Sequential()
# model.add(layers.Flatten())  # 第二维展平 784个元素组成的一维数组
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))  # 多分类问题

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_images, train_labels,
          epochs=5, batch_size=32,
          validation_data=(test_images, test_labels),
          validation_freq=1)

model.summary()
