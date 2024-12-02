import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


print("timer 5s")
time.sleep(5)
print(torch.cuda.is_available())
print("start")

iris = load_iris()
X = iris.data
Y = iris.target

X = torch.FloatTensor(X)
Y = torch.LongTensor(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
input_size = 4
hidden_size = 20
output_size = 3
model = Model(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100

for epoch in range(num_epochs):
    
    
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        
        
with torch.no_grad():
    # 在测试集上进行预测
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    correct = (predicted == Y_test).sum().item()
    total = Y_test.size(0)
    accuracy = correct / total * 100
    print(f'Accuracy of the model on the test set: {accuracy}%')