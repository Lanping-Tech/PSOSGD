import argparse

import torch

from psosgd_dataset import load_CIFAR10
from psosgd_trainer import PSOSGD_Trainer
from psosgd_trainer import PSOSGD_Trainer_Config as TG
from models.ResNet import Config as MG
from psosgd_optimizer import Config as OG

def main(args):

    train_loader, test_loader = load_CIFAR10(args.train_batch_size, args.test_batch_size)


    model_config = MG()
    optimizer_config = OG()
    trainer_config = TG(model_config, optimizer_config)
    trainer = PSOSGD_Trainer(trainer_config)

    loss_fn = torch.nn.CrossEntropyLoss()

    trainer.train(train_loader, loss_fn, args.epochs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--train_batch_size', default=100, type=int, help='training batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='testing batch size')
    args = parser.parse_args()

    main(args)