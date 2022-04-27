from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model_save_path = './checkpoint/mnist.ckpt'
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)
preNum = int(input("input the number of test pictures:"))

for i in range(preNum):
    image_path = input("the path of test picture:")
    img = Image.open(image_path)

    image = plt.imread(image_path)
    plt.set_cmap('gray')
    plt.imshow(image)

    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    # 将输入图片变为只有黑色和白色的高对比图片
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:  # 小于200的变为纯黑色
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0  # 其余变为纯白色

    # 由于神经网络训练时都是按照batch输入
    # 为了满足神经网络输入特征的shape(图片总数，宽，高）
    # 所以要将28行28列的数据[28,28]二维数据---变为--->一个28行28列的数据[1,28,28]三维数据
    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]  # 插入一个维度
    result = model.predict(x_predict)  #  返回前向传播计算结果，预测
    pred = tf.argmax(result, axis=1)

    print('\n')
    tf.print(pred)

    plt.pause(1)  # 相当于plt.show()，但是只显示1秒
    plt.close()