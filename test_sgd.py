
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch

import diabetes_dataset
from model import NeuralNetwork


import numpy

batch_size = 8
epochs = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create data loaders.
train_dataloader = diabetes_dataset.load(batch_size, shuffle=True)

# ------------------梯度下降法（结合动量）-------------------#
def sgd_train(dataloader, epochs=epochs, device=device):
    # Define model
    model = NeuralNetwork().double().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-1, momentum=0.5, dampening=0.5)

    fitness = []
    for t in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fitness.append(loss.item())
    return fitness

# ---------------粒子群+梯度下降法（结合动量）--------------------#

def sgd_train(dataloader, epochs=epochs, device=device):
    # Define model
    model = NeuralNetwork().double().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-1, momentum=0.5)

    fitness = []
    for t in range(epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fitness.append(loss.item())
    return fitness

