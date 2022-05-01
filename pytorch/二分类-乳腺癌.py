from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt

"""
包含了威斯康辛州记录的569个病人的乳腺癌恶性/良性(1/0)类别型数据，以及与之对应的30个维度的生理指标数据
"""

# 1 准备数据
(x_train, y_train) = datasets.load_breast_cancer(return_X_y=True)
x_train = torch.tensor(x_train).to(torch.float32)
y_train = torch.tensor(y_train).to(torch.float32).unsqueeze(1)

# 2 建立模型
class Model(torch.nn.Module):
    """Module会自动根据计算图自动反向传播"""
    def __init__(self):
        super(Model, self).__init__()
        """包含权重w和偏置b两个tensor，(30, 15)是指输入x和输出y的特征维度(数据库中的字段)"""
        self.linear1 = torch.nn.Linear(30, 16)
        self.linear2 = torch.nn.Linear(16, 8)
        self.linear3 = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))  # y hat
        return x

model = Model()

# 3 构造损失函数和优化器
criterion = torch.nn.BCELoss(size_average=True)  # 逻辑回归
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 自动完成参数的初始化操作

# 4 训练过程：forward, backward, update
epoch_list = []
loss_list = []
for epoch in range(2000):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    print(epoch, loss.item())  # loss是计算图
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播自动计算梯度
    optimizer.step()  # update 参数w b

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()