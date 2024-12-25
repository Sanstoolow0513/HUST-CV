import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 数据预处理与加载
# 将图像数据转换为张量，并进行归一化处理，方便模型训练
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 下载并加载训练集，设置 train=True，下载测试集时 train=False
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 数据加载器，设置批量大小，用于按批次向模型输入数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 卷积层 1：输入通道 1（MNIST 图像是灰度图），输出通道 16，卷积核大小 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  
        self.relu1 = nn.ReLU()
        # 最大池化层，池化核大小 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2)  
        # 卷积层 2：输入通道 16，输出通道 32，卷积核大小 3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 全连接层，将卷积后的特征图展平后输入，输出 10 个神经元，对应 10 个数字类别
        self.fc1 = nn.Linear(32 * 5 * 5, 10)  

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 展平特征图，为输入全连接层做准备
        x = x.view(-1, 32 * 5 * 5)  
        x = self.fc1(x)
        return x

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')

# 可视化预测结果（选做）
# 随机选一批测试数据展示预测结果
images, labels = next(iter(test_loader))
outputs = model(images)
_, predicted = torch.max(outputs, 1)
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].numpy().squeeze(), cmap='gray')
    ax.set_title(f'Predicted: {predicted[i].item()}')
    ax.axis('off')
plt.show()  




