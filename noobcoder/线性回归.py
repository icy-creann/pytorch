"""
数据准备
定义模型
定义损失函数和优化器
训练模型：前向传播、计算损失、清空梯度、反向传播、更新参数
预测、评估模型
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声

# 打印部分数据
print(X[:5])
print(Y[:5])


import torch.nn as nn

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)  # 输入维度2，输出维度1
    
    def forward(self, x):
        return self.linear(x)  # 前向传播，返回预测结果

# 创建模型实例
model = LinearRegressionModel()

# 损失函数（均方误差）
criterion = nn.MSELoss()

# 优化器（使用 SGD 或 Adam）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)  # 学习率设置为0.01,动量设置为0.9

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()  # 切换到训练模式

    # 前向传播
    predictions = model(X)

    # 计算损失
    loss = criterion(predictions.squeeze(), Y)  # predictions.squeeze() 是为了匹配输出维度
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    # 打印损失
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 查看训练后的权重和偏置
print(f'Predicted weight: {model.linear.weight.data.numpy()}')
print(f'Predicted bias: {model.linear.bias.data.numpy()}')

# 在新数据上做预测
with torch.no_grad():  # 评估时不需要计算梯度
    predictions = model(X)

# 可视化预测与实际值
plt.scatter(X[:, 0], Y, color='blue', label='True values')
plt.scatter(X[:, 0], predictions, color='red', label='Predictions')
plt.legend()
plt.show()