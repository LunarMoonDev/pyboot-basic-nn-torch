# importing libraries
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    '''
        Basic neural network model with two hidden layers

        @param in_features: number of input features
        @param h1: number of nodes for hidden layer 1
        @param h2: number of nodes for hidden layer 2
        @out_features: number of output features
    '''

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)   # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer

    def forward(self, x):
        '''
            connects the initialized neural layers with the given input
            
            @param x: training input
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
