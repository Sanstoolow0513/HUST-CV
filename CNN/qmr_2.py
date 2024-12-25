import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 输出10类

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


# 数据加载与处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
train_subset_length = int(len(train_dataset) * 0.1)
train_subset_indices = torch.randperm(len(train_dataset))[:train_subset_length]
train_subset = torch.utils.data.Subset(train_dataset, train_subset_indices)
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)
# 这里假设使用10%的测试数据，实际应用中可按需求调整
test_subset_length = int(len(test_dataset) * 0.1)
test_subset_indices = torch.randperm(len(test_dataset))[:test_subset_length]
test_subset = torch.utils.data.Subset(test_dataset, test_subset_indices)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1, shuffle=True)


# 初始化模型、损失函数与优化器
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")


class SameDigitClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(SameDigitClassifier, self).__init__()
        self.base_model = pretrained_model

    def forward(self, img1, img2):
        # 分别预测
        output1 = self.base_model(img1)
        output2 = self.base_model(img2)

        # 获取每张图片的预测类别
        pred1 = torch.argmax(output1, dim=1)
        pred2 = torch.argmax(output2, dim=1)

        # 判断是否是同一类别
        same = (pred1 == pred2).float()
        return same


same_digit_model = SameDigitClassifier(model)
correct_count = 0
total_count = 0
for (img1, label1), (img2, label2) in zip(test_loader, test_loader):
    img1, img2 = img1.to('cpu'), img2.to('cpu')
    output = same_digit_model(img1, img2)
    total_count += 1
    if (label1 == label2) == (output.item() == 1.0):
        correct_count += 1
    print(f"Label1: {label1.item()}, Label2: {label2.item()}, Same Digit: {output.item()}")

accuracy = correct_count / total_count
print(f"Test accuracy: {accuracy}")
