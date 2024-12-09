import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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



# 数据预处理（和之前保持一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集（这里仅示例加载测试集，可按需加载训练集等）
test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 导入保存的模型参数
model_state_dict = torch.load('mnist_cnn_model.pth', map_location=device)
model.load_state_dict(model_state_dict)
model.eval()  # 将模型切换到评估模式

# 此时模型已经恢复到保存时的状态，可以进行后续的测试等操作
# 例如以下是简单的对测试集中的部分数据进行预测示例（可按需调整）
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1).item()
        print(f"预测结果: {pred}, 真实标签: {target.item()}")