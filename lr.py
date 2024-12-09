import torch
import matplotlib.pyplot as plt

torch.manual_seed(1024)

x = torch.linspace(100,300,200)

x = (x - torch.mean(x)) / torch.std(x)

epsilon = torch.randn(x.shape)
y = x*10 + 5 + epsilon

plt.scatter(x,y)