import torch
from torch.optim import Optimizer

import random

class Config:

    """优化器配置参数"""
    def __init__(self):
        self.lr = 1e-3
        self.momentum = 0.5
        self.dampening = 0.5
        self.weight_decay = 0
        self.nesterov = False
        self.weight_gradient = 0.5
        self.vlimit_max = 0.5
        self.vlimit_min = -0.5
        self.weight_particle_optmized_location = 0.33
        self.weight_global_optmized_location = 0.33

class PSOSGD(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, weight_gradient=0.00005, 
                 vlimit_max = 0.5, vlimit_min = -0.5, weight_particle_optmized_location = 0.33,
                 weight_global_optmized_location = 0.33, **kwargs):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, weight_gradient=weight_gradient,
                        vlimit_max = vlimit_max, vlimit_min = vlimit_min, weight_particle_optmized_location = weight_particle_optmized_location,
                        weight_global_optmized_location = weight_global_optmized_location)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PSOSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PSOSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, local_best_param_group, global_best_param_group, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_gradient = group['weight_gradient']
            vlimit_max = group['vlimit_max']
            vlimit_min = group['vlimit_min']
            weight_particle_optmized_location = group['weight_particle_optmized_location']
            weight_global_optmized_location = group['weight_global_optmized_location']

            for p_index, p in enumerate(group['params']):
                local_best_p = local_best_param_group[p_index]
                global_best_p = global_best_param_group[p_index]
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        buf.mul_(momentum)
                        buf.add_(local_best_p.sub(p), alpha=weight_particle_optmized_location * random.random())
                        buf.add_(global_best_p.sub(p), alpha=weight_global_optmized_location * random.random())
                        buf.add_(d_p, alpha=weight_gradient)

                        buf[buf > vlimit_max] = vlimit_max
                        buf[buf < vlimit_min] = vlimit_min
     
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss
