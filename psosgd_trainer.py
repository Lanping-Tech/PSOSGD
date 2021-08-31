import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import importlib

import numpy as np

import torch

from psosgd_optimizer import PSOSGD

import matplotlib.pyplot as plt

from tqdm import tqdm

class PSOSGD_Trainer_Config:

    """训练配置参数"""
    def __init__(self,
                 model_config,
                 optimizer_config,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 n_particle = 5,
                 output_path = 'output', **kwargs):

        # 预留模型参数
        self.model_config = model_config

        # 优化器参数
        self.optimizer_config = optimizer_config

        self.device = device

        self.n_particle = n_particle

        self.output_path = output_path


class PSOSGD_Trainer:

    def __init__(self, config: PSOSGD_Trainer_Config):
        self.config = config
        model_lib = importlib.import_module(config.model_config.model_lib)

        self.models= [model_lib.Model(**config.model_config.__dict__).double().to(self.config.device) for _ in range(self.config.n_particle)]

        self.optimizers = [PSOSGD(model.parameters(), **config.optimizer_config.__dict__) for model in self.models]

    def train(self, data_loader, loss_fn, epochs):

        loss_fn = loss_fn.double().to(self.config.device)

        losses = []
        eval_losses = []
        eval_accs = []
        local_best_param_groups = [(float('inf'), None) for _ in range(self.config.n_particle)]
        global_best_param_group = (float('inf'), None)

        for t in range(epochs):
            # 切换训练状态
            print('Epoch ' + str(t))
            for model in self.models:
                model.train()

            for (X, y) in tqdm(data_loader):
                X, y = X.double().to(self.config.device), y.to(self.config.device)
                batch_losses = []
                for model, optimizer in zip(self.models, self.optimizers):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                    optimizer.zero_grad()
                    loss.backward()
                    
                    batch_losses.append(loss.item())

                for i in range(self.config.n_particle):
                    if local_best_param_groups[i][0] > batch_losses[i]:
                        local_best_param_groups[i] = (batch_losses[i], [torch.clone(param).detach() for param in self.models[i].parameters()])

                    if global_best_param_group[0] > batch_losses[i]:
                        global_best_param_group = (batch_losses[i], [torch.clone(param).detach() for param in self.models[i].parameters()])

                for i in range(self.config.n_particle):
                    self.optimizers[i].step(local_best_param_groups[i][1], global_best_param_group[1])

                losses.append(batch_losses)

            print('Eval ' + str(t))

            eval_loss, eval_acc = self.test(data_loader, loss_fn)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            best_model_index = np.argmax(eval_acc)

            print('Best Loss model: {} {}'.format(np.argmax(eval_loss), max(eval_loss)))
            print('Best ACC model: {} {}'.format(np.argmax(eval_acc), max(eval_acc)))

            model_save_path = os.path.join(self.config.output_path, 'epoch-{}-model-{}.pth'.format(t, best_model_index))
            self.save_model(self.models[best_model_index], model_save_path)

        self.performance_display(np.array(losses).T, 'Train_Loss')
        self.performance_display(np.array(eval_losses).T, 'Eval_Loss')
        self.performance_display(np.array(eval_accs).T, 'Eval_ACC')

        return losses, eval_losses, eval_accs

    def test(self, data_loader, loss_fn):

        for model in self.models:
            model.eval()
        
        loss_fn = loss_fn.double().to(self.config.device)

        losses = []
        accs = []

        with torch.no_grad():
            for model in self.models:
                test_loss = 0
                test_acc = 0
                total = 0
                batch_num = 0
                for (data, target) in tqdm(data_loader):
                    data, target = data.double().to(self.config.device), target.to(self.config.device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    test_loss += loss.item()
                    prediction = torch.max(output, 1)
                    total += target.size(0)
                    test_acc += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
                    batch_num += 1

                losses.append(test_loss / batch_num)
                accs.append(test_acc / total)

        return losses, accs

    def save_model(self, model, path):
        torch.save(model, path)
        print("Checkpoint saved to {}".format(path))

    def performance_display(self, metric_value, metric_name):
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for model_index in range(self.config.n_particle):
            color_index = model_index % len(color)
            plt.plot(list(range(1, len(metric_value[model_index])+1)), 
                     metric_value[model_index], 
                     color=color[color_index], 
                     linewidth=1.5, 
                     label='Model {}'.format(model_index))

        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel(metric_name)
        plt.grid(linestyle='--')
        fig_path = os.path.join(self.config.output_path,metric_name+'.png')
        plt.savefig(fig_path, dpi=500, bbox_inches = 'tight')
        plt.cla()




