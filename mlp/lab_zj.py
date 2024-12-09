import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

print("timer 1s")
time.sleep(1)
print("start")
print("IF GPU READY:",torch.cuda.is_available())

df = pd.read_csv('dataset.csv')
df = df.sample(frac=1).reset_index(drop=True) # random.shuffle(indices)
gaussian_data = df.iloc[:,:2].to_numpy()
labels = df.iloc[:,2].to_numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
X = torch.FloatTensor(gaussian_data).to(device)
Y = torch.LongTensor(labels).to(device)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
Y_test -= 1
Y_train -= 1

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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
    
input_size = 2
hidden_size = 32
output_size = 4
model = Model(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 40
for epoch in range(num_epochs):
    
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

with torch.no_grad():
    # test_outputs = model(X_test)
    # _, predicted = torch.max(test_outputs.data, 1)
    # correct = (predicted == Y_test).sum().item()
    # total = Y_test.size(0)
    # accuracy = correct / total * 100
    # print(f'Accuracy of the model on the test set: {accuracy}%')
    
    # # 遍历的准确率
    # model.eval()
    # correct = 0
    # total = 0
    # for inputs, labels in test_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = model(inputs)
    #     _, predicted = torch.max(outputs, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # accuracy = 100 * correct / total
    # print(f'Accuracy of the model on the test set: {accuracy}%')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.append(predicted)
        all_labels.append(labels)
    
    # 合并所有批次的预测和标签
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # 计算准确率
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy}%')