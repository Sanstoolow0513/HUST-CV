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

# 获取卷积层的输出
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

# 获取激活图
activations = conv_output.squeeze().detach().cpu()  # 形状是 [64, 8, 8]

# 打印激活图的统计信息
print(f"Activations shape: {activations.shape}")
print(f"Activations min: {activations.min().item()}, max: {activations.max().item()}, mean: {activations.mean().item()}")

# Score-CAM 计算
weights = []
for i, activation in enumerate(activations):
    # 对激活图进行上采样，使其尺寸与输入图像一致
    activation = torch.nn.functional.interpolate(
        activation.unsqueeze(0).unsqueeze(0),  # 增加 batch 和 channel 维度
        size=(28, 28),  # 上采样到 28x28
        mode='bilinear',  # 双线性插值
        align_corners=False
    ).squeeze()  # 去掉多余的维度

    # 将激活图与输入图像相乘
    masked_image = image * activation.to(device)

    # 前向传播计算得分
    output = model(masked_image)
    score = output[:, target_class].item()
    weights.append(score)

    # 打印每个激活图的得分
    print(f"Activation {i}: score = {score}")

# 打印权重的统计信息
weights = np.array(weights)
print(f"Weights shape: {weights.shape}")
print(f"Weights min: {weights.min()}, max: {weights.max()}, mean: {weights.mean()}")

# 计算加权激活图
heatmap = np.sum(weights[:, None, None] * activations.numpy(), axis=0)

# 打印热力图的统计信息
print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}, mean: {heatmap.mean()}")

# 对热力图进行上采样，使其尺寸与输入图像一致
heatmap = torch.nn.functional.interpolate(
    torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float(),  # 增加 batch 和 channel 维度
    size=(28, 28),  # 上采样到 28x28
    mode='bilinear',  # 双线性插值
    align_corners=False
).squeeze().numpy()

# 归一化热力图到 [0, 1] 范围
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)

# 可视化热力图
plt.matshow(heatmap, cmap='viridis')
plt.title('Score-CAM Heatmap')
plt.colorbar()
plt.show()