import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

# 选取10%的数据作为测试集（按照你的要求调整测试集范围）
num_test_samples = len(test_dataset)
subset_size = int(num_test_samples * 0.1)
subset_indices = random.sample(range(num_test_samples), subset_size)
subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)

test_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

# 初始化模型、将模型部署到合适设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
# 这里假设模型已经训练好（可以加载之前保存的模型参数等方式来恢复训练好的状态）
# 示例中先不涉及重新训练模型的过程，若没训练过需要先训练好模型

# 定义判断两张图片数字是否相同的函数
def check_same_digit(image1, image2):
    output1 = model(image1.unsqueeze(0).to(device))
    output2 = model(image2.unsqueeze(0).to(device))
    pred1 = torch.argmax(output1, dim=1).item()
    pred2 = torch.argmax(output2, dim=1).item()
    return 1 if pred1 == pred2 else 0


# 从测试数据加载器中获取两张不同的图片进行测试（简单示例，可以多次循环等更灵活操作）
image_iter = iter(test_loader)
image1, _ = next(image_iter)
image2, _ = None, None
while True:
    try:
        image2, _ = next(image_iter)
        if not torch.equal(image1, image2):  # 确保两张图片不是同一张
            break
    except StopIteration:
        image_iter = iter(test_loader)

result = check_same_digit(image1, image2)
print(f"两张图片对应的数字是否相同：{result}")


# 定义用于提取图片特征的 CNN 网络
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义基于特征提取的 Siamese 网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor()

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        # 使用欧氏距离作为距离度量
        distance = torch.sqrt(torch.sum((feat1 - feat2) ** 2, dim=1))
        # 映射到0-1范围作为是否相同的概率
        output = nn.Sigmoid()(1 - distance.unsqueeze(1))
        return output


# 初始化模型、损失函数和优化器
model2 = SiameseNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)

# 训练和评估模型2
train_losses2 = []
test_losses2 = []
train_accuracies2 = []
test_accuracies2 = []

for epoch in range(10):
    model2.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (img1, img2, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model2(img1, img2)
        loss = criterion(output, label.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses2.append(train_loss)
    train_accuracies2.append(train_accuracy)

    model2.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            output = model2(img1, img2)
            loss = criterion(output, label.unsqueeze(1))

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    test_losses2.append(test_loss)
    test_accuracies2.append(test_accuracy)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
