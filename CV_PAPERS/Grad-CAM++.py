import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from source import CNN  # 导入你的CNN模型

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device), strict=False)  # 修复警告
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载图像
image = Image.open('example.png').convert('L')  # 读取灰度图像
image = transform(image).unsqueeze(0).to(device)

# 获取卷积层的输出和梯度
conv_output = None
def hook_fn(module, input, output):
    global conv_output
    conv_output = output

# 注册钩子到最后一个卷积层
model.conv2.register_forward_hook(hook_fn)

# 前向传播
output = model(image)
target_class = output.argmax(dim=1).item()
print(f"Target class: {target_class}")

# 反向传播计算梯度
output[:, target_class].backward()

# 获取梯度
gradients = model.conv2.weight.grad  # 形状是 [64, 32, 5, 5]
activations = conv_output.detach()  # 形状是 [1, 64, 8, 8]

# 打印梯度和激活值的统计信息
print(f"Gradients shape: {gradients.shape}")
print(f"Gradients min: {gradients.min().item()}, max: {gradients.max().item()}, mean: {gradients.mean().item()}")
print(f"Activations shape: {activations.shape}")
print(f"Activations min: {activations.min().item()}, max: {activations.max().item()}, mean: {activations.mean().item()}")

# Grad-CAM++ 计算
# 1. 计算 alpha
# 对每个输出通道的梯度求和，得到形状为 [64, 1, 1]
alpha = torch.sum(gradients, dim=(2, 3), keepdim=True)  # 形状是 [64, 32, 1, 1]
alpha = torch.mean(alpha, dim=1, keepdim=True)  # 对输入通道取平均，形状是 [64, 1, 1, 1]
alpha = torch.nn.functional.relu(alpha)  # 只保留正梯度
alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-10)  # 归一化

# 2. 调整 alpha 的形状以匹配 activations
alpha = alpha.permute(1, 0, 2, 3)  # 形状变为 [1, 64, 1, 1]

# 3. 检查形状是否匹配
if alpha.shape[1] != activations.shape[1]:
    raise ValueError(f"Alpha shape {alpha.shape} does not match activations shape {activations.shape}")

# 4. 计算加权激活
weighted_activations = torch.sum(activations * alpha, dim=1, keepdim=True)

# 5. 计算热力图
#heatmap = torch.nn.functional.relu(weighted_activations)  # 只保留正值
heatmap = weighted_activations.squeeze().cpu().numpy()

# 6. 打印热力图的统计信息
print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}, mean: {heatmap.mean()}")

# 7. 归一化热力图
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)

# 可视化热力图
plt.matshow(heatmap, cmap='viridis')
plt.title('Grad-CAM++ Heatmap')
plt.colorbar()
plt.show()