import os
import copy
import librosa as li
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union

from tqdm import tqdm

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
import IPython.display as ipd


def play_audiomnist(x):
    return ipd.Audio(x.detach().cpu().numpy().flatten(), rate=16000)  # load a NumPy array


def plot_audiomnist(x: torch.Tensor, y: Union[int, torch.Tensor], model: nn.Module):
    """
    Given an audio waveform, ground-truth label, and a classification model:
      * plot audio
      * plot model's prediction for audio (vector of class scores)

    :param x: a tensor holding an audio waveform to classify and plot
    :param y: an integer or integer tensor holding the ground-truth class label
    :param model: a classification model for generating predictions
    """

    device = x.device  # hold onto original device

    x = x.clone().detach().cpu()
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().cpu().item()

    # use model to compute class scores and predicted label
    y_scores = torch.nn.functional.softmax(
        model(x.reshape(1, 1, 16000).to(device)), dim=-1
    ).detach().cpu()
    y_pred = y_scores.argmax()

    # initialize plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    width = 0.5
    linewidth = 2.0

    # image plot
    axs[0].plot(x.squeeze().numpy(), 'k-')
    axs[0].set_xlabel('Sample idx')
    axs[0].set_ylabel('Amplitude')

    # class scores plot
    axs[1].bar(
        list(range(0, 10)),
        y_scores.flatten().detach().cpu().numpy(),
        width,
        color='black',
        label='class scores',
        edgecolor='black',
        linewidth=linewidth
    )

    # formatting
    fig.suptitle(f"True Label: {y}, Predicted Label: {y_pred}", y=1.1)
    axs[1].grid(False)
    axs[1].spines['left'].set_linewidth(linewidth)
    axs[1].set_xlim(-1, 10)
    axs[1].tick_params(bottom=True, left=True)
    axs[1].set_yscale('log')
    axs[1].set_xticks(list(range(0, 10)))
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()


def plot_mnist(x: torch.Tensor, y: Union[int, torch.Tensor], model: nn.Module):
    """
    Given a grayscale image, ground-truth label, and a classification model:
      * plot image
      * plot model's prediction for image (vector of class scores)

    :param x: a tensor holding an image to classify and plot
    :param y: an integer or integer tensor holding the ground-truth class label
    :param model: a classification model for generating predictions
    """

    device = x.device  # hold onto original device

    x = x.clone().detach().cpu()
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().cpu()

    # use model to compute class scores and predicted label
    y_scores = torch.nn.functional.softmax(
        model(x.reshape(1, 1, 28, 28).to(device)), dim=-1
    ).detach().cpu()
    y_pred = y_scores.argmax()

    # initialize plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    width = 0.5
    margin = 0.0025
    linewidth = 2.0

    # image plot
    axs[0].imshow(x.squeeze().numpy(), cmap='gray')

    # class scores plot
    axs[1].bar(
        list(range(0, 10)),
        y_scores.flatten().detach().cpu().numpy(),
        width,
        color='black',
        label='class scores',
        edgecolor='black',
        linewidth=linewidth
    )

    # formatting
    fig.suptitle(f"True Label: {y}, Predicted Label: {y_pred}", y=1.1)
    axs[1].grid(False)
    axs[1].spines['left'].set_linewidth(linewidth)
    axs[1].set_xlim(-1, 10)
    axs[1].tick_params(bottom=True, left=True)
    axs[1].set_yscale('log')
    axs[1].set_xticks(list(range(0, 10)))
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()






