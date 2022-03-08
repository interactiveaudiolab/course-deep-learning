import io
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def make_grid(batch: torch.Tensor, size: int, title: str = "Training Images"):
    """
    Plot images in a grid with the specified dimensions.
    """

    # check that a square grid of the given size can be plotted
    assert batch.shape[0] >= size * size

    images = batch[:size * size, ...].detach().cpu()

    fig = plt.figure(figsize=(size, size))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(
            vutils.make_grid(images, padding=2, normalize=True),
            (1, 2, 0)
        )
    )

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def make_loss_plot(max_epochs: int, loss_d: np.ndarray, loss_g: np.ndarray):
    """
    Plot discriminator and generator losses by epoch
    """
    fig = plt.figure()
    plt.xlim((0, max_epochs))
    plt.plot(range(0, max_epochs), loss_d, label='discriminator')
    plt.plot(range(0, max_epochs), loss_g, label='generator')
    plt.legend()

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))


def make_score_plot(max_epochs: int, scores_real: np.ndarray, scores_fake: np.ndarray):
    """
    Plot discriminator scores for real and generated inputs
    """
    fig = plt.figure()
    plt.xlim((0, max_epochs))
    plt.plot(range(0, max_epochs), scores_real, label='real')
    plt.plot(range(0, max_epochs), scores_fake, label='generated')
    plt.legend()

    # save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)

    # return plot as image
    return ToTensor()(np.array(img))
