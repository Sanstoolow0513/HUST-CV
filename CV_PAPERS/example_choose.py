import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dataset = datasets.MNIST(root=current_dir, train=False, download=True, transform=transform)

# 选择指定索引的图片（例如索引为10的图片）
index = 10  # 你可以修改这个索引
image, label = test_dataset[index]

# 将张量转换为PIL图像并保存
image = transforms.ToPILImage()(image)
image.save('example.png')

print(f"Saved image with index {index} (label: {label}) as example.png")