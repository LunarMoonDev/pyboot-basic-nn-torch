# import libraries
import time

import numpy as np
import torch
import torch.nn as nn
from numpy import genfromtxt
from torch.utils.data import DataLoader

from model import Model

# constants
X_TRAIN = './data/processed/x_train.csv'
Y_TRAIN = './data/processed/y_train.csv'
RANDOM_SEED = 4
LEARNING_RATE = 0.01
VERSION = '0.0'

# data prep
X_train = genfromtxt(X_TRAIN, delimiter=',', dtype=np.float)
y_train = genfromtxt(Y_TRAIN, delimiter=',', dtype=np.float)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)


trainloader = DataLoader(X_train, batch_size=60, shuffle=True)

# model prep
torch.manual_seed(RANDOM_SEED)
model = Model()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training the model
epochs = 100
losses = []

for i in range(epochs):
    i += 1
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)

    if i % 10 == 1:         # reduces the output log per 10 epochs
        print(f'epoch: {i: 2} loss: {loss.item(): 10.8f}')
        # can do checkpoint saves here also
    optimizer.zero_grad()   # resets the gradients
    loss.backward()
    optimizer.step()

# saving the model
torch.save(model.state_dict(),
           f'./models/model.S.{VERSION}.{int(time.time())}.pt')
