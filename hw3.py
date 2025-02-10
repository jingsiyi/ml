import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

X = np.array([
    [0, 0, 0, 0, 0, 0, 0.697, 0.460],
    [1, 0, 1, 0, 0, 0, 0.774, 0.376],
    [1, 0, 0, 0, 0, 0, 0.634, 0.264],
    [0, 0, 1, 0, 0, 0, 0.608, 0.318],
    [2, 1, 0, 0, 1, 1, 0.556, 0.215],
    [0, 1, 0, 1, 1, 1, 0.403, 0.237],
    [1, 1, 0, 1, 1, 1, 0.481, 0.149],
    [1, 1, 0, 0, 1, 0, 0.437, 0.211],
    [1, 1, 1, 1, 1, 0, 0.666, 0.091],
    [0, 3, 2, 0, 2, 1, 0.243, 0.267],
    [2, 3, 2, 2, 2, 0, 0.245, 0.057],
    [2, 0, 0, 2, 1, 1, 0.343, 0.099],
    [0, 0, 0, 1, 0, 0, 0.639, 0.161],
    [2, 0, 1, 0, 0, 0, 0.657, 0.198],
    [1, 1, 1, 0, 0, 1, 0.360, 0.370],
    [2, 0, 1, 2, 0, 1, 0.593, 0.042],
    [0, 0, 1, 1, 1, 0, 0.719, 0.103]
])

y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

class Net(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, h_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def train_bp(model, X_train, y_train, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_acc_bp(model, X_train, y_train, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        if (epoch + 1) % X_train.shape[0] == 0:
            optimizer.step()
            optimizer.zero_grad()

def cross_val(model_class, train_func, X_t, y_t, k=17):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_t)):
        model = model_class(in_size=8, h_size=5, out_size=1)

        X_train, y_train = X_t[train_idx], y_t[train_idx]
        X_test, y_test = X_t[test_idx], y_t[test_idx]

        train_func(model, X_train, y_train)

        with torch.no_grad():
            preds = model(X_test).round()
            acc = accuracy_score(y_test.numpy(), preds.numpy())
            accs.append(acc)

    avg_acc = np.mean(accs)
    print(f'平均准确率: {avg_acc:.4f}')
    return avg_acc


print("标准BP:")
bp_acc = cross_val(Net, train_bp, X_t, y_t)

print("\n累积BP:")
acc_bp_acc = cross_val(Net, train_acc_bp, X_t, y_t)
