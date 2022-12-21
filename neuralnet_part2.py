# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.neuralNet = nn.Sequential(nn.Dropout(0.5), nn.Linear(1568, 150), nn.ELU(), nn.Linear(150, 180), nn.ELU(), nn.Dropout(0.5), nn.Linear(180, 200), nn.ELU(), nn.Linear(200, out_size))
        self.optimiser = optim.SGD(self.neuralNet.parameters(), self.lrate, momentum=0.9, weight_decay=0.001)
        #Out channel 12 seems to be correct number
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, out_channels=12, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(3, 18, affine=True),
            nn.ELU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(12, out_channels=20, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(3, 24, affine=True),
            nn.MaxPool2d(2,2),
            nn.ELU())
        # NOTE: kernel size increase, batchnorm, increase out channel not help
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.MaxPool2d(5, stride = 1, padding=2),
            # nn.BatchNorm2d(29),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.GroupNorm(3, 3, affine=True),
            nn.ELU())
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        out = self.layer1(x.view(x.shape[0], 3, 31, 31))

        out = self.layer2(out)

        out = self.layer3(out)
        # out = self.layer4(out)
        # print(out.shape)
        # print(out.shape)

        x = out.reshape(out.size(0), -1)
        return self.neuralNet(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        self.optimiser.zero_grad()
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimiser.step()
        return loss.item()

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    lossFunction = nn.CrossEntropyLoss()
    learningRate = 0.01
    mean = train_set.mean()
    std = train_set.std()
    train_set = (train_set - train_set.mean()) / (train_set.std())
    training = get_dataset_from_arrays(train_set, train_labels)
    trainingData = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=False)
    neuralNet = NeuralNet(learningRate, lossFunction, len(train_set[0]), 4)
    lossResult = []
    for i in range(epochs):
        totalLoss = 0
        for j in trainingData:
            totalLoss += neuralNet.step(j['features'], j['labels'])
        lossResult.append(totalLoss)
    neuralNet = neuralNet.eval()
    dev_set = (dev_set - mean) / (std)
    prediction = neuralNet(dev_set).detach().numpy()
    yhats = np.argmax(prediction, axis=1)
    return lossResult, yhats, neuralNet
