import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 选择10%的数据
train_size = int(len(mnist_train) * 0.1)
test_size = int(len(mnist_test) * 0.1)
mnist_train, _ = random_split(mnist_train, [train_size, len(mnist_train) - train_size])
mnist_test, _ = random_split(mnist_test, [test_size, len(mnist_test) - test_size])

class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # 获取原始数据集的标签并转换为Python整数列表
        self.labels = [self.dataset.dataset.targets[i].item() for i in self.dataset.indices]
        self.indices_by_label = {label: [] for label in range(10)}
        for idx, label in enumerate(self.labels):
            self.indices_by_label[label].append(idx)

    def __getitem__(self, index):
        image1, label1 = self.dataset[index]
        # 确保label1是一个Python整数
        label1 = label1.item() if isinstance(label1, torch.Tensor) else label1
        
        # 50%几率选择相同标签，50%几率选择不同标签
        should_get_same_class = np.random.randint(2)
        if should_get_same_class:
            while True:
                idx2 = np.random.choice(self.indices_by_label[label1])
                if idx2 != index:
                    break
            image2, _ = self.dataset[idx2]
            target = torch.tensor(1., dtype=torch.float)
        else:
            label2 = np.random.choice([l for l in range(10) if l != label1])
            idx2 = np.random.choice(self.indices_by_label[label2])
            image2, _ = self.dataset[idx2]
            target = torch.tensor(0., dtype=torch.float)
        
        return image1, image2, target

    def __len__(self):
        return len(self.dataset)
    

siamese_train_dataset = SiameseDataset(mnist_train)
siamese_test_dataset = SiameseDataset(mnist_test)

batch_size = 64
train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 3 * 3, 128)

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

model = SiameseNetwork()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def train_epoch(model, data_loader, criterion, optimizer, device='cpu'):
    model.train()
    running_loss = 0.0
    for i, (img1, img2, labels) in enumerate(data_loader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device=device)
    print(f'Training Loss: {train_loss:.4f}')
    
def compute_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, labels in data_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            euclidean_distance = F.pairwise_distance(output1, output2)
            predictions = euclidean_distance < 0.5
            correct += torch.sum(predictions == labels.byte()).item()
            total += labels.size(0)
    return correct / total

accuracy = compute_accuracy(model, test_loader, device)
print(f'Accuracy on test set: {accuracy:.2%}')



