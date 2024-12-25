import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# 选择 10% 数据
train_size = int(0.1 * len(train_dataset))
test_size = int(0.1 * len(test_dataset))
train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset) - train_size])
test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])

# 构造样本对
class PairsDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data, self.targets = dataset.data, dataset.targets

    def __getitem__(self, index):
        # 随机选择一个正样本对
        x1, y1 = self.data[index], self.targets[index]
        positive_index = torch.randint(0, len(self.data), (1,)).item()
        while self.targets[positive_index] != y1:
            positive_index = torch.randint(0, len(self.data), (1,)).item()
        x2 = self.data[positive_index]
        label = torch.tensor(1, dtype=torch.float32)

        # 随机选择一个负样本对
        if torch.randint(0, 2, (1,)).item() == 0:
            negative_index = torch.randint(0, len(self.data), (1,)).item()
            while self.targets[negative_index] == y1:
                negative_index = torch.randint(0, len(self.data), (1,)).item()
            x2 = self.data[negative_index]
            label = torch.tensor(0, dtype=torch.float32)

        return x1.unsqueeze(0).float() / 255.0, x2.unsqueeze(0).float() / 255.0, label

    def __len__(self):
        return len(self.dataset)

train_pairs = PairsDataset(train_dataset)
test_pairs = PairsDataset(test_dataset)

train_loader = DataLoader(train_pairs, batch_size=128, shuffle=True)
test_loader = DataLoader(test_pairs, batch_size=128, shuffle=False)


# 定义用于单张图片分类的 CNN 网络
class SingleImageClassifier(nn.Module):
    def __init__(self):
        super(SingleImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义基于分类结果比较的网络
class CompareByClassification(nn.Module):
    def __init__(self):
        super(CompareByClassification, self).__init__()
        self.classifier = SingleImageClassifier()

    def forward(self, img1, img2):
        out1 = self.classifier(img1)
        out2 = self.classifier(img2)
        # 使用余弦相似度作为比较依据
        cosine_sim = nn.CosineSimilarity(dim=1, eps=0.000001)
        similarity = cosine_sim(out1, out2)
        # 映射到0-1范围作为是否相同的概率
        output = nn.Sigmoid()(similarity.unsqueeze(1))
        return output


# 初始化模型、损失函数和优化器
model1 = CompareByClassification()
criterion = nn.BCELoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# 训练和评估模型1
train_losses1 = []
test_losses1 = []
train_accuracies1 = []
test_accuracies1 = []

for epoch in range(10):
    model1.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (img1, img2, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model1(img1, img2)
        loss = criterion(output, label.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses1.append(train_loss)
    train_accuracies1.append(train_accuracy)

    model1.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            output = model1(img1, img2)
            loss = criterion(output, label.unsqueeze(1))

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total
    test_losses1.append(test_loss)
    test_accuracies1.append(test_accuracy)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
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
