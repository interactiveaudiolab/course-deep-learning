import torch
import math
import decimal
from torch import nn


class CNN(nn.Module):
    """Adaptation of AudioNet (arXiv:1807.03418)."""

    def __init__(self,
                 input_dim=16000,  # 1s @ 16kHz
                 n_classes=10      # 10 digits
                 ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 100, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv1d(100, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2))

        # compute necessary dimensions of final linear layer
        conv_shape = self._compute_output_size(input_dim)
        self.fc = nn.Linear(conv_shape, n_classes)

    def _compute_output_size(self, input_dim):
        x = torch.zeros((1, 1, input_dim))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
        return x.numel()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class AudioNet(nn.Module):
    """
    Wrapper for AudioNet waveform convolutional model proposed in Becker et al.
    (https://arxiv.org/abs/1807.03418), with normalization preprocessing. Code
    adapted from Adversarial Robustness Toolbox (https://tinyurl.com/54sdatn3)
    """

    def __init__(self,
                 input_dim: int = 16000,
                 n_classes: int = 10,
                 normalize: bool = True,
                 ):
        super().__init__()

        self.normalize = normalize

        self.cnn = CNN(
            input_dim=input_dim,
            n_classes=n_classes
        )

    def load_weights(self, path: str):
        """
        Load weights from checkpoint file
        """

        # check if file exists
        if not path or not os.path.isfile(path):
            return

        try:
            self.cnn.load_state_dict(torch.load(path))
        except RuntimeError:
            self.load_state_dict(torch.load(path))

    def forward(self, x: torch.Tensor):

        if self.normalize:
            x = (1.0 / torch.max(
                torch.abs(x) + 1e-8, dim=-1, keepdim=True
            )[0]) * x * 0.95

        return self.cnn(x)

    @staticmethod
    def match_predict(y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Determine whether target pairs are equivalent
        """
        if y_pred.ndim >= 2 and y_pred.shape[-1] >= 2:
            y_pred = torch.argmax(y_pred, dim=-1)
        else:
            y_pred = torch.round(y_pred.to(torch.float32))

        if y_true.ndim >= 2 and y_true.shape[-1] >= 2:
            y_true = torch.argmax(y_true, dim=-1)
        else:
            y_true = torch.round(y_true.to(torch.float32))

        return y_pred == y_true
