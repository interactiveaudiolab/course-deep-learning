import os

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


def load_mnist(batch_size: int = 128):
    """
    Load MNIST dataset.
    :return: DataLoader object
    """

    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    # format image data, but do not normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # download MNIST data
    data = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    # load MNIST data
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        **cuda_kwargs
    )

    return loader


def load_fashionmnist(batch_size: int = 128):
    """
    Load MNIST dataset.
    :return: DataLoader object
    """

    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    # format image data, but do not normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # download MNIST data
    data = datasets.FashionMNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )

    # load MNIST data
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        **cuda_kwargs
    )

    return loader


def load_celeba(batch_size: int = 128):
    """
    Load CelebA dataset.
    :return: DataLoader object
    """

    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    # crop data to 3x64x64 images (channels/width/height)
    image_size = 64

    # format image data
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # download MNIST data
    data = datasets.CelebA(
        './data',
        split='all',
        download=True,
        transform=transform
    )

    # load MNIST data
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        **cuda_kwargs
    )

    return loader
