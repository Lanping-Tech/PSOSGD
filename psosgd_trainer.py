import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import importlib

import torch

class PSOSGD_Trainer:

    def __init__(self, 
                 model_name='Linear',
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 number_particle=150):
        self.model_name = model_name
        model_lib = importlib.import_module('models.' + model_name)
        self.device = device
        self.number_particle = number_particle
        self.models= [model_lib.Model().double().to(device) for _ in range(number_particle)]

        self.optimizers = [torch.optim.SGD(model.parameters(), lr=2e-1, momentum=0.5, dampening=0.5) for model in self.models]

    def train(self, data_loader, loss_fn, epochs):
        fitness = []
        for t in range(epochs):
            for batch, (X, y) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)
                temp_fitness = []
                for model, optimizer in zip(self.models, self.optimizers):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    

                    temp_fitness.append(loss.item())
                    # print(loss.item())
                
                print(temp_fitness)
                print(min(temp_fitness))
                print()

                fitness.append(min(temp_fitness))
        return fitness


