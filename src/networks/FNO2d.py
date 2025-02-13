"""
FNO2D.py
-----------

Implementation of a 2D Fourier Neural Operator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List
from src.networks.spectral import (
    SpectralConv2d,
)  # Annahme: spectral.py ist im gleichen Verzeichnis


class FNOBlock2d(nn.Module):
    """
    A 2D Fourier Neural Operator (FNO) block that applies spectral convolution
    followed by a bypass convolution and an activation function.

    Parameters
    ----------
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels.
        modes (int):
            Number of Fourier modes to use in the spectral convolution.
        activation (callable):
            Activation function to apply after the convolutions.
        spectral_hidden_layers (int, optional):
            Number of hidden layers in the spectral MLP. Default is 2.
        spectral_hidden_width (int, optional):
            Width of the hidden layers in the spectral MLP. Default is 4.

    Methods
    ----------
        forward(x):
            Applies the spectral convolution, bypass convolution, and activation function to the input tensor x.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        activation,
        spectral_hidden_layers=2,
        spectral_hidden_width=4,
    ):
        super(FNOBlock2d, self).__init__()
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes,
            n_layers=spectral_hidden_layers,
            hidden_width=spectral_hidden_width,
        )
        self.bypass_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,  # Kernelgröße ist eins
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral_conv(x) + self.bypass_conv(x))


class FNO2d(nn.Module):
    """
    A 2D Fourier Neural Operator (FNO) model.

    Parameters
    ----------
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels.
        modes (int):
            Number of Fourier modes to use.
        width (int):
            Width of the network (number of channels in the hidden layers).
        activation (callable):
            Activation function to use.
        n_blocks (int, optional):
            Number of FNO blocks to use. Default is 4.
        spectral_hidden_layers (int, optional):
            Number of hidden layers in the spectral MLP. Default is 2.
        spectral_hidden_width (int, optional):
            Width of the hidden layers in the spectral MLP. Default is 4.

    Attributes
    ----------
        lifting (nn.Conv2d): Convolutional layer to lift the input to the desired width.
        fno_blocks (nn.ModuleList): List of FNO blocks.
        projection (nn.Conv2d): Convolutional layer to project the output to the desired number of channels.

    Methods
    ----------
        forward(x):
            Forward pass of the FNO2d model.

            Args:
                x (torch.Tensor):
                    Input tensor of shape (batch_size, in_channels, height, width).

            Returns:
                torch.Tensor:
                    Output tensor of shape (batch_size, out_channels, height, width).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        width,
        activation,
        n_blocks=4,
        spectral_hidden_layers=2,
        spectral_hidden_width=4,
    ):
        super(FNO2d, self).__init__()
        self.lifting = nn.Conv2d(
            in_channels,
            width,
            kernel_size=1,
        )
        self.fno_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.fno_blocks.append(
                FNOBlock2d(
                    width,
                    width,
                    modes,
                    activation,
                    spectral_hidden_layers,
                    spectral_hidden_width,
                )
            )
        self.projection = nn.Conv2d(
            width,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        x = self.lifting(x)
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        x = self.projection(x)
        return x
