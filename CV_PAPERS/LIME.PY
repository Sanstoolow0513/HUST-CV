import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from source import CNN  # 导入你的CNN模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn_model.pth', map_location=device), strict=False)  # 修复警告
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载图像
image = Image.open('example.png').convert('L')  # 读取灰度图像
image = transform(image).unsqueeze(0).to(device)

# LIME解释器
def batch_predict(images):
    model.eval()
    # 将输入图像从3通道RGB转换回1通道灰度
    batch = torch.stack([transform(Image.fromarray((img[:, :, 0] * 255).astype(np.uint8)).convert('L')) for img in images], dim=0).to(device)
    logits = model(batch)
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

# 将单通道图像转换为3通道图像
image_np = image.squeeze().cpu().numpy()
image_rgb = np.stack([image_np] * 3, axis=-1)  # 复制单通道为3通道

# 创建LIME解释器
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image_rgb, batch_predict, top_labels=5, hide_color=0, num_samples=1000)

# 可视化解释结果
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.title('LIME Explanation')
plt.show()