from torch.nn import Module, ReLU, Conv2d, MaxPool2d
from torch.nn.init import kaiming_uniform_

class CNN(Module):
    def __init__(self,n_channels):#self表示类的实例本身属性，n_channels表示输入图像的通道数
        super().__init__()#调用父类Module的构造函数
        
        # 输入层到卷积层1
        self.hidden1 = Conv2d(n_channels, 32, (3,3))#接收n_channels个通道（如RGB图像的3个通道），输出32个输出通道，卷积核大小3x3
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')#使用Kaiming均匀初始化方法初始化卷积层的权重，并指定ReLU作为非线性激活函数
        self.act1 = ReLU()#创建ReLU激活函数实例
        
        # 池化层1
        self.pool1 = MaxPool2d((2,2), stride=(2,2))#创建一个2x2的最大池化层，步幅为2x2