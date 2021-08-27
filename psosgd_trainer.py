import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import importlib

import numpy as np

import torch

from psosgd_optimizer import PSOSGD

class PSOSGD_Trainer_Config:

    """训练配置参数"""
    def __init__(self,
                 model_config,
                 optimizer_config,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 n_particle = 5):

        # 预留模型参数
        self.model_config = model_config

        # 优化器参数
        self.optimizer_config = optimizer_config

        self.device = device

        self.n_particle = n_particle


class PSOSGD_Trainer:

    def __init__(self, config: PSOSGD_Trainer_Config):
        self.config = config
        model_lib = importlib.import_module(config.model_config.model_lib)

        self.models= [model_lib.Model(**config.model_config.__dict__).double().to(self.config.device) for _ in range(self.config.n_particle)]

        self.optimizers = [PSOSGD(model.parameters(), **config.optimizer_config.__dict__) for model in self.models]

    def train(self, data_loader, loss_fn, epochs):

        loss_fn = loss_fn.double().to(self.config.device)

        for model in self.models:
            model.train()

        fitness = []
        local_best_param_groups = [(float('inf'), None) for _ in range(self.config.n_particle)]
        global_best_param_group = (float('inf'), None)
        for t in range(epochs):
            for batch, (X, y) in enumerate(data_loader):
                X, y = X.double().to(self.config.device), y.to(self.config.device)
                temp_fitness = []
                for model, optimizer in zip(self.models, self.optimizers):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    
                    temp_fitness.append(loss.item())
                    # print(loss.item())
                
                print(min(temp_fitness))
                # print(np.argmin(np.array(temp_fitness)))
                # print()

                for i in range(self.config.n_particle):
                    if local_best_param_groups[i][0] > temp_fitness[i]:
                        local_best_param_groups[i] = (temp_fitness[i], [torch.clone(param).detach() for param in self.models[i].parameters()])

                    if global_best_param_group[0] > temp_fitness[i]:
                        global_best_param_group = (temp_fitness[i], [torch.clone(param).detach() for param in self.models[i].parameters()])

                # print(local_best_param_groups)
                # print(global_best_param_group)

                for i in range(self.config.n_particle):
                    self.optimizers[i].step(local_best_param_groups[i][1], global_best_param_group[1])

                fitness.append(min(temp_fitness))

        return fitness


