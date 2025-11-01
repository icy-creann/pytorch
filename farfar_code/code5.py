from torch.nn import Module, ReLU, Conv2d, MaxPool2d, Linear,Softmax
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class CNN(Module):
    def __init__(self,n_channels):#self表示类的实例本身属性，n_channels表示输入图像的通道数
        super().__init__()#调用父类Module的构造函数

        # 输入层到卷积层1
        self.hidden1 = Conv2d(n_channels, 32, (3,3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')# 初始化卷积层1的权重
        self.act1 = ReLU()# 激活函数1，ReLU
        self.pool1 = MaxPool2d((2,2), stride=(2,2))# 池化层1，2x2最大池化，步长为2

        # 卷积层2,进一步提取卷积层1输出的特征图中的特征
        self.hidden2 = Conv2d(32, 32, (3,3))# 输入32个通道(与卷积层1输出通道数相同)，输出32个通道，卷积核大小3x3
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')# 初始化卷积层2的权重
        self.act2 = ReLU()# 激活函数2，ReLU
        self.pool2 = MaxPool2d((2,2), stride=(2,2))# 池化层2，2x2最大池化，步长为2


        #第一个全连接层
        self.hidden3 = Linear(5*5*32,100)#输入特征数5*5*32(池化层2输出的特征图大小为5x5，通道数为32)，输出特征数100
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')# 初始化全连接层3的权重
        self.act3 = ReLU()# 激活函数3，ReLU

        #第二个全连接层
        self.hidden4 = Linear(100,10)#输入特征数100，输出特征数10（对应10个类别）
        xavier_uniform_(self.hidden4.weight)# 初始化全连接层4的权重
        self.act4 = Softmax(dim=1)# 激活函数4，Softmax，沿着类别维度进行归一化


    def forward(self, X):
        # 输入层到卷积层1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)

        # 卷积层2
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)

        # 展平特征图
        X = X.view(-1, 5*5*32)

        # 第一个全连接层
        X = self.hidden3(X)
        X = self.act3(X)

        # 第二个全连接层
        X = self.hidden4(X)
        X = self.act4(X)
        return X