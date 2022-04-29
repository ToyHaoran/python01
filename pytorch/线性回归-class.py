import torch

# 准备数据 x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    """
    线性模型：Module会自动根据计算图自动反向传播；
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        """包含权重w和偏置b两个tensor，(1,1)是指输入x和输出y的特征维度"""
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# 构造损失函数和优化器
criterion = torch.nn.MSELoss(size_average=False, reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),  # 自动完成参数的初始化操作
                            lr=0.01)

# 训练过程：forward, backward, update
for epoch in range(200):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # backward 自动计算梯度
    optimizer.step()  # update 参数w b

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)