import torch
from torch.optim import Optimizer

import random

class Config:

    """优化器配置参数"""
    def __init__(self,
                 lr = 1e-1, 
                 momentum = 0.5,
                 dampening=0.5,
                 weight_decay = 0,
                 nesterov = False,
                 vlimit_max = 0.5,
                 vlimit_min = -0.5,
                 xlimit_max = 10,
                 xlimit_min = -10,
                 weight_particle_optmized_location = 0.33,
                 weight_global_optmized_location = 0.33, **kwargs):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.vlimit_max = vlimit_max
        self.vlimit_min = vlimit_min
        self.xlimit_max = xlimit_max
        self.xlimit_min = xlimit_min
        self.weight_particle_optmized_location = weight_particle_optmized_location
        self.weight_global_optmized_location = weight_global_optmized_location

class PSOSGD(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,vlimit_max = 0.5, vlimit_min = -0.5, 
                 xlimit_max = 10, xlimit_min = -10,
                 weight_particle_optmized_location = 0.33,
                 weight_global_optmized_location = 0.33, **kwargs):

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        vlimit_max = vlimit_max, vlimit_min = vlimit_min, 
                        xlimit_max = xlimit_max, xlimit_min = xlimit_min, 
                        weight_particle_optmized_location = weight_particle_optmized_location,
                        weight_global_optmized_location = weight_global_optmized_location)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PSOSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PSOSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, local_best_param_group, global_best_param_group, is_psosgd, closure=None):
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
            vlimit_max = group['vlimit_max']
            vlimit_min = group['vlimit_min']
            xlimit_max = group['xlimit_max']
            xlimit_min = group['xlimit_min']
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
                        if is_psosgd:
                            buf.sub_(local_best_p.sub(p), alpha=weight_particle_optmized_location * random.random())
                            buf.sub_(global_best_p.sub(p), alpha=weight_global_optmized_location * random.random())
                        buf.add_(d_p, alpha=1-dampening)

                        if is_psosgd:
                            buf[buf > vlimit_max] = vlimit_max
                            buf[buf < vlimit_min] = vlimit_min
     
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)
                # p[p>xlimit_max] = xlimit_max
                # p[p<xlimit_min] = xlimit_min

        return loss
