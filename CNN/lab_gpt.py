import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# 1. 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 2. 从训练集和测试集选取10%数据
train_subset = torch.utils.data.Subset(train_data, np.random.choice(len(train_data), size=int(0.1 * len(train_data)), replace=False))
test_subset = torch.utils.data.Subset(test_data, np.random.choice(len(test_data), size=int(0.1 * len(test_data)), replace=False))

# 3. 自定义数据集，生成图片对
class MNISTPairDataset(Dataset):
    def __init__(self, mnist_data):
        self.data = mnist_data
        self.labels = mnist_data.targets
        self.pairs = []
        self.targets = []

        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                # 比较两张图片是否是相同的数字
                if self.labels[i] == self.labels[j]:
                    self.pairs.append((self.data[i][0], self.data[j][0]))  # 同一类，标签为1
                    self.targets.append(1)
                else:
                    self.pairs.append((self.data[i][0], self.data[j][0]))  # 不同类，标签为0
                    self.targets.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.targets[idx]
        return torch.cat((pair[0], pair[1]), dim=0), label

# 4. 创建 DataLoader
train_loader = DataLoader(MNISTPairDataset(train_subset), batch_size=32, shuffle=True)
test_loader = DataLoader(MNISTPairDataset(test_subset), batch_size=32, shuffle=False)

# 5. 定义卷积神经网络（Siamese 网络）
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

# 6. 训练模型
model = SiameseNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练循环
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

# 7. 测试模型
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

print(f'Test Accuracy: {100 * correct / total:.2f}%')
