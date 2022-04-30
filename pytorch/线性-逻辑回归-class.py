import torch

# 1 准备数据
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
from matplotlib import pyplot as plt

x_data = torch.tensor([[1.0], [2.0], [3.0]])
# y_data = torch.tensor([[2.0], [4.0], [6.0]])  # 线性回归
y_data = torch.tensor([[0.], [0.], [1.]])  # 二分类 逻辑回归

# 2 设计模型
class LogisticModel(torch.nn.Module):
    """
    Module会自动根据计算图自动反向传播
    线性模型：linear
    逻辑斯谛回归：增加了激活函数sigmoid
    """
    def __init__(self):
        super(LogisticModel, self).__init__()
        """包含权重w和偏置b两个tensor，(1,1)是指输入x和输出y的特征维度"""
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # y_pred = self.linear(x)  # 线性回归
        y_pred = torch.sigmoid(self.linear(x))  # 逻辑回归
        return y_pred

model = LogisticModel()

# 3 构造损失函数和优化器
# criterion = torch.nn.MSELoss(size_average=False, reduction='sum')  # 线性回归
criterion = torch.nn.BCELoss(size_average=False)  # 逻辑回归
optimizer = torch.optim.SGD(model.parameters(),  # 自动完成参数的初始化操作
                            lr=0.01)

# 4 训练过程：forward, backward, update
epoch_list = []
loss_list = []
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    print(epoch, loss.item())  # loss是计算图
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # backward 反向传播自动计算梯度
    optimizer.step()  # update 参数w b

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()