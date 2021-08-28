import torch
import torch.nn as nn

import torchvision


class Config:

    """
    模型配置参数

    You can use the following models officially provided by torchvision：

    'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 
    'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1',
    'inception_v3', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'googlenet',
    'mobilenet_v2', 'mobilenet_v3', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
    """
    def __init__(self, 
                 model_name = 'resnet18',
                 model_pretrained = False,
                 n_classes = 10, **kwargs):
        self.model_lib = 'models.VisionModels'
        self.model_name = model_name
        self.pretrained = model_pretrained
        self.n_classes = n_classes


class Model(nn.Module):

    def __init__(self, model_name, pretrained, n_classes, **kwargs):
        super().__init__()

        model_func = getattr(torchvision.models, model_name)
        self.target_model = model_func(pretrained=pretrained, num_classes=n_classes)

    def forward(self, x):
        output = self.target_model(x)
        return output





