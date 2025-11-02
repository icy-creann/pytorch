import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

if __name__ == '__main__':

    # # 示例数据
    # X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
    # Y_data = [1, 0, 1, 0]  # 目标标签

    # # 创建数据集实例
    # dataset = MyDataset(X_data, Y_data)

    dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
    
    X_data = dataset.iloc[:,2 :-1].values  # 所有行，除了最后一列
    Y_data = dataset.iloc[:, -1].values  # 所有行，最后一列

    # # 热编码处理
    # encoder = OneHotEncoder()
    # X_data = encoder.fit_transform(X_data).toarray()


    # 创建数据集实例
    dataset = MyDataset(X_data, Y_data)

    # 创建 DataLoader 实例，batch_size 设置每次加载的样本数量
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 打印加载的数据
    for epoch in range(2):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}:')
            print(f'Inputs: {inputs}')
            print(f'Labels: {labels}')
