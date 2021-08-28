import os

import argparse

import torch

import psosgd_dataset
from psosgd_trainer import PSOSGD_Trainer
from psosgd_trainer import PSOSGD_Trainer_Config as TG
from models.VisionModels import Config as MG
from psosgd_optimizer import Config as OG

def main(args):

    train_loader, test_loader = getattr(psosgd_dataset, 'load_'+args.dataset_name)(args.train_batch_size, args.test_batch_size)

    model_config = MG(**args.__dict__)
    optimizer_config = OG(**args.__dict__)
    trainer_config = TG(model_config, optimizer_config, **args.__dict__)
    trainer = PSOSGD_Trainer(trainer_config)

    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.train(train_loader, loss_fn, args.epochs)
    test_loss, test_acc = trainer.test(test_loader, loss_fn)

    print('All model loss on test dataset: '+test_loss)
    print('All model ACC on test dataset: '+test_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")

    # model
    parser.add_argument('--model_name', default='resnet18', type=str, help='particle model name')
    parser.add_argument('--model_pretrained', default=False, type=bool, help='particle model name')
    parser.add_argument('--n_classes', default=10, type=int, help='number of classes')

    # dataset
    parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='dataset name')
    parser.add_argument('--train_batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size')

    # opitimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.5, type=float, help='momentum')
    parser.add_argument('--dampening', default=0.5, type=float, help='dampening')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight_decay')
    parser.add_argument('--nesterov', default=False, type=bool, help='nesterov')
    parser.add_argument('--vlimit_max', default=0.5, type=float, help='vlimit_max')
    parser.add_argument('--vlimit_min', default=-0.5, type=float, help='vlimit_min')
    parser.add_argument('--xlimit_max', default=10, type=float, help='xlimit_max')
    parser.add_argument('--xlimit_min', default=-10, type=float, help='xlimit_min')
    parser.add_argument('--weight_particle_optmized_location', default=0.33, type=float, help='weight_particle_optmized_location')
    parser.add_argument('--weight_global_optmized_location', default=0.33, type=float, help='weight_global_optmized_location')

    # trainer
    parser.add_argument('--divice', default="cuda" if torch.cuda.is_available() else "cpu", type=str, help='divice')
    parser.add_argument('--n_particle', default=10, type=int, help='number of particle')
    parser.add_argument('--output_path', default="output", type=str, help='output path')


    parser.add_argument('--epochs', default=200, type=int, help='number of epochs tp train for')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)