# import libaries
import torch
import torch.nn as nn
import numpy as np
from model import Model
from torch.utils.data import DataLoader

# constants
X_TEST = './data/processed/x_test.csv'
Y_TEST = './data/processed/y_test.csv'
MODEL = './models/model.S.0.0.1664527317.pt'

# data prep
X_test = np.genfromtxt(X_TEST, delimiter=',', dtype=np.float)
y_test = np.genfromtxt(Y_TEST, delimiter=',', dtype=np.float)

X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

testloader = DataLoader(X_test, batch_size=60, shuffle=False)

# model prep
torch.manual_seed(4)
model = Model()
model.load_state_dict(torch.load(MODEL))
model.eval()

# loss function
criterion = nn.CrossEntropyLoss()

# test the model
with torch.no_grad():   # doesn't cache gradients
    y_val = model.forward(X_test)
    loss = criterion(y_val, y_test)

print(f'Test loss: {loss:.8f}')

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i + 1:2}. {str(y_val):38} [{y_test[i]}]')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'\n{correct} out of {len(y_test)} = {100 * correct/ len(y_test):.2f}% correct\n')

# unseen data
mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])
labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor', 'Mystery iris']

with torch.no_grad():
    print("Unseen data: \n", model(mystery_iris))
    print(labels[model(mystery_iris).argmax().item()])
