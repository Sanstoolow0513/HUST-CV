import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

# 抽取 10%的数据
train_indices = np.random.choice(len(train_dataset), int(0.1 * len(train_dataset)), replace=False)
test_indices = np.random.choice(len(test_dataset), int(0.1 * len(test_dataset)), replace=False)

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
test_subset = torch.utils.data.Subset(test_dataset, test_indices)

# 数据对生成类
class MNISTPairDataset(Dataset):
    def __init__(self, dataset, same_ratio=0.5):
        self.dataset = dataset
        self.same_ratio = same_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image1, label1 = self.dataset[index]

        # 以一定比例生成相同数字的图片对
        if np.random.rand() < self.same_ratio:
            valid_indices = np.where(np.array(self.dataset.dataset.targets) == label1)[0]
            if len(valid_indices) > 0:  # 添加边界检查
                same_index = np.random.choice(valid_indices, 1)[0]
                image2, label2 = self.dataset[same_index]
                label = 1
        else:
            valid_indices = np.where(np.array(self.dataset.dataset.targets)!= label1)[0]
            if len(valid_indices) > 0:  # 添加边界检查
                different_index = np.random.choice(valid_indices, 1)[0]
                image2, label2 = self.dataset[different_index]
                label = 0

        return image1, image2, label

# 定义用于单张图片分类的 CNN 网络
class SingleImageClassifier(nn.Module):
    def __init__(self):
        super(SingleImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 基于分类网络构建用于比较两张图片数字是否相同的网络
class ImageComparator(nn.Module):
    def __init__(self, classifier):
        super(ImageComparator, self).__init__()
        self.classifier = classifier

    def forward(self, image1, image2):
        output1 = self.classifier(image1)
        output2 = self.classifier(image2)
        # 比较两个分类结果
        return torch.eq(torch.argmax(output1, dim=1), torch.argmax(output2, dim=1)).float()

# 定义训练和评估函数
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for image1, image2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(image1, image2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for image1, image2, labels in test_loader:
                outputs = model(image1, image2)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

        test_loss /= len(test_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_losses, test_losses, train_accuracies, test_accuracies

# 数据加载
train_pair_dataset = MNISTPairDataset(train_subset)
test_pair_dataset = MNISTPairDataset(test_subset)

train_loader = DataLoader(train_pair_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_pair_dataset, batch_size=64, shuffle=False)

# 模型实例化、损失函数和优化器定义
classifier = SingleImageClassifier()
model = ImageComparator(classifier)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练和评估
epochs = 10
train_losses, test_losses, train_accuracies, test_accuracies = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs)

# 绘制损失和准确率曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.show()