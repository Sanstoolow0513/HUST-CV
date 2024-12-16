import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 1. 定义卷积神经网络（Siamese Network）
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        # 计算欧式距离
        distance = torch.abs(output1 - output2)
        output = self.sigmoid(distance)
        return output

    # 剪枝功能：对最后一层卷积进行剪枝
    def prune_conv2(self, test_loader, K):
        # 假设 K 是剪枝的特征图数量
        self.eval()
        
        # 计算输出特征图的激活值（平均值）
        activations = []
        with torch.no_grad():
            for data, target in test_loader:
                img1, img2 = data[:, 0:1, :, :], data[:, 1:2, :, :]
                output1 = self.forward_one(img1)
                output2 = self.forward_one(img2)
                
                # 计算输出的平均激活值
                activation = torch.mean(torch.abs(output1), dim=[0, 2, 3])  # 对M×N维度求平均
                activations.append(activation)
                
                activation = torch.mean(torch.abs(output2), dim=[0, 2, 3])  # 对M×N维度求平均
                activations.append(activation)
        
        activations = torch.stack(activations)
        avg_activations = activations.mean(dim=0)  # 对所有样本取平均
        # 对激活值排序
        _, sorted_indices = torch.sort(avg_activations)
        
        # 剪枝最小激活值的特征图
        prune_indices = sorted_indices[:K]
        prune_mask = torch.ones_like(avg_activations)
        prune_mask[prune_indices] = 0
        
        # 对最后一层卷积层的权重进行剪枝
        with torch.no_grad():
            self.conv2.weight.data[:, prune_indices, :, :] = 0
            self.conv2.bias.data[prune_indices] = 0
        
        return prune_mask

# 2. 创建数据加载器和数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 3. 实例化模型、定义优化器和损失函数
model = SiameseNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        img1, img2 = data[:, 0:1, :, :], data[:, 1:2, :, :]
        target = target.float()
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output.view(-1), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

# 5. 在测试集上评估模型性能
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        img1, img2 = data[:, 0:1, :, :], data[:, 1:2, :, :]
        output = model(img1, img2)
        predicted = (output.view(-1) > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy before pruning: {100 * correct / total:.2f}%')

# 6. 剪枝操作
K = 1  # 选择剪枝的数量
prune_mask = model.prune_conv2(test_loader, K)

# 7. 重新评估剪枝后的网络
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        img1, img2 = data[:, 0:1, :, :], data[:, 1:2, :, :]
        output = model(img1, img2)
        predicted = (output.view(-1) > 0.5).float()
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy after pruning: {100 * correct / total:.2f}%')
