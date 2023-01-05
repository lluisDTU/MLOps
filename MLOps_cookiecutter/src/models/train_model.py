import argparse
import os
import sys

import click
import helper
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim

repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root+'src/data/')
import make_dataset
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    trainloader,_ = Obtain_Train_Test_Data()

    epochs = 5

    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f"Training loss: {running_loss / len(trainloader)}")
            print(f'Accuracy: {accuracy.item() * 100}%')

cli.add_command(train)

if __name__ == "__main__":
    cli()