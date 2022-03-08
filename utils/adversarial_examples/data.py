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


def load_mnist(train_batch_size: int = 64, test_batch_size: int = 1000):
    """
    Load MNIST dataset. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    :return: train and test DataLoader objects
    """

    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    # format image data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download MNIST data
    train_data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.MNIST(
        './data',
        train=False,
        transform=transform,
    )

    # load MNIST data
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        **cuda_kwargs
    )

    return train_loader, test_loader


def load_audiomnist(data_dir, train_batch_size: int = 64, test_batch_size: int = 128):

    audio_list = sorted(list(Path(data_dir).rglob(f'*.wav')))
    cache_list = sorted(list(Path(data_dir).rglob('*.pt')))  # check for cached dataset

    if len(cache_list) > 0:
        tx = torch.load(os.path.join(data_dir, 'audiomnist_tx.pt'))
        ty = torch.load(os.path.join(data_dir, 'audiomnist_ty.pt'))

    else:
        tx = torch.zeros((len(audio_list), 1, 16000))
        ty = torch.zeros(len(audio_list), dtype=torch.long)

        pbar = tqdm(audio_list, total=len(audio_list))

        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading AudioMNIST ({os.path.basename(audio_fn)})')
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=16000,
                                  duration=1.0)
            waveform = torch.from_numpy(waveform)

            tx[i, :, :waveform.shape[-1]] = waveform
            ty[i] = int(os.path.basename(audio_fn).split("_")[0])

        torch.save(tx, os.path.join(data_dir, 'audiomnist_tx.pt'))
        torch.save(ty, os.path.join(data_dir, 'audiomnist_ty.pt'))

    # partition data
    tx_train, ty_train, tx_test, ty_test = [], [], [], []
    for i in range(10):

        idx = ty == i
        tx_i = tx[idx]
        ty_i = ty[idx]

        split = int(0.8 * len(tx_i))

        tx_train.append(tx_i[:split]), ty_train.append(ty_i[:split])
        tx_test.append(tx_i[split:]), ty_test.append(ty_i[split:])

    tx_train = torch.cat(tx_train, dim=0)
    ty_train = torch.cat(ty_train, dim=0)
    tx_test = torch.cat(tx_test, dim=0)
    ty_test = torch.cat(ty_test, dim=0)

    # create datasets
    train_data = torch.utils.data.TensorDataset(tx_train, ty_train)

    test_data = torch.utils.data.TensorDataset(tx_test, ty_test)

    # load data
    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        **cuda_kwargs
    )

    return train_loader, test_loader
