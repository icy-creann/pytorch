import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt



class NumberDetector(nn.Module):
    def __init__(self,device = 'cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.log_softmax(self.fc4(x), dim=1)
        return x
    
    def get_dataloader(self):
        to_tensor = transforms.Compose([transforms.ToTensor()])
        dataset = MNIST(root='./datasets', train=True, download=True, transform=to_tensor)
        return DataLoader(dataset, batch_size=64, shuffle=True) 
    

    def accuracy(self,test_data):
        correct = 0
        total = 0
        with torch.no_grad():
            for (x,y) in test_data:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.forward(x.view(-1, 28*28))
                for i,output in enumerate(outputs):
                    if torch.argmax(output) == y[i]:
                        correct += 1
                    total += 1
        return correct / total
    


if __name__ == '__main__':
    #定义模型，加载数据
    epochs = 3  #训练轮数
    net = NumberDetector(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))    #定义模型，使用cuda加速
    train_data = net.get_dataloader()   #获取训练数据
    test_data = net.get_dataloader()   #获取测试数据
    

    #训练模型
    print("epoch:",0,";accuracy:",net.accuracy(test_data))      #获取初始准确率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)       #定义优化器
    for epoch in range(epochs):
        for (x,y) in train_data:                                #遍历训练数据
            x = x.to(net.device)                                #将数据移动到cuda上
            y = y.to(net.device)                                #将标签移动到cuda上
            net.zero_grad()                                     #将模型参数的梯度清零                                   
            output = net.forward(x.view(-1, 28*28))             #前向传播
            loss = nn.functional.nll_loss(output, y)            #计算损失
            loss.backward()                                     #反向传播
            optimizer.step()                                    #更新模型参数
        print("epoch:",epoch+1,";accuracy:",net.accuracy(test_data))     #获取当前准确率

    for (n,(x, _)) in enumerate(test_data):                 #遍历测试数据
        if n > 3:                                           #只显示前4张图片
            break   
        x = x.to(net.device)                                #将数据移动到cuda上
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))    #获取模型预测结果

        plt.figure(n)
        plt.imshow(x[0].cpu().view(28,28))
        plt.title("predict:"+str(predict.item()))
    plt.show()