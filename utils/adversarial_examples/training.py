import os
import copy
import librosa as li
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
import IPython.display as ipd


def train_audiomnist(model,
                     device,
                     train_loader,
                     test_loader,
                     epochs: int = 14,
                     save_model: bool = True
                     ):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=1.0
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):

        # track loss
        training_loss = 0.0
        validation_loss = 0

        # track accuracy
        correct = 0
        total = 0

        pbar = tqdm(train_loader, total=len(train_loader))

        model.train()
        for batch_idx, batch_data in enumerate(pbar):

            pbar.set_description(
                f'Epoch {epoch + 1}, batch {batch_idx + 1}/{len(train_loader)}')

            inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # sum training loss
            training_loss += loss.item()

        model.eval()
        with torch.no_grad():

            pbar = tqdm(test_loader, total=len(test_loader))
            for batch_idx, batch_data in enumerate(pbar):

                pbar.set_description(
                    f'Validation, batch {batch_idx + 1}/{len(test_loader)}')

                inputs, labels = batch_data

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # sum validation loss
                validation_loss += loss.item()

                # calculate validation accuracy
                preds = torch.max(outputs.data, 1)[1]

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        # calculate final metrics
        validation_loss /= len(test_loader)
        training_loss /= len(train_loader)
        accuracy = 100 * correct / total

        # if best model thus far, save
        if accuracy > best_acc and save_model:
            print(f"New best accuracy: {accuracy}; saving model")
            best_model = copy.deepcopy(model.state_dict())
            best_acc = accuracy
            torch.save(
                best_model,
                "audionet.pt"
            )

        # update step-size scheduler
        scheduler.step()


def test_audiomnist(model, device, test_loader):

    model.to(device)
    model.eval()

    # track accuracy
    correct = 0
    total = 0

    with torch.no_grad():

        pbar = tqdm(test_loader, total=len(test_loader))
        for batch_idx, batch_data in enumerate(pbar):

            pbar.set_description(
                f'Validation, batch {batch_idx + 1}/{len(test_loader)}')

            inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # calculate validation accuracy
            preds = torch.max(outputs.data, 1)[1]

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nModel accuracy: {accuracy}")


def train_mnist(
        model,
        device,
        train_loader,
        test_loader,
        epochs: int = 14,
        log_interval: int = 50,
        save_model: bool = True
):
    """
    Train a simple MNIST classifier. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    :param model:
    :param device:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param log_interval:
    :param save_model:

    :return:
    """

    # configure optimization
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):

        # training step
        model.train()  # training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # validation step
        model.eval()  # evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "../models/mnist_cnn.pt")


def test_mnist(model, device, test_loader):
    """
    Evaluate a simple MNIST classifier. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    model.eval()  # evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))