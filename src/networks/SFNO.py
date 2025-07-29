"""
SFNO.py
-----------

Implementation of a 2D Spherical Fourier Neural Operator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List
from torch_harmonics import RealSHT, InverseRealSHT

class SHT(nn.Module):
    """A wrapper for the Spherical Harmonics transform 

    Allows to call it with an interface similar to that of FFT
    """
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self._SHT_cache = nn.ModuleDict()
        self._iSHT_cache = nn.ModuleDict()

    def sht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, height, width = x.shape # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                modes_width = height // 2
            else:
                modes_width = height
            modes_height = height
        else:
            modes_height, modes_width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            sht = self._SHT_cache[cache_key]
        except KeyError:
            
            sht = (
                RealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm
                )
                .to(dtype=self.dtype)
                .to(device=x.device)
            )
            self._SHT_cache[cache_key] = sht
        
        return sht(x)


    def isht(self, x, s=None, norm="ortho", grid="equiangular"):
        *_, modes_height, modes_width = x.shape # height = latitude, width = longitude
        if s is None:
            if grid == "equiangular":
                width = modes_width*2
            else:
                width = modes_width
            height = modes_height
        else:
            height, width = s

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"

        try:
            isht = self._iSHT_cache[cache_key]
        except KeyError:
            isht = (
                InverseRealSHT(
                    nlat=height,
                    nlon=width,
                    lmax=modes_height,
                    mmax=modes_width,
                    grid=grid,
                    norm=norm
                )
                .to(dtype=self.dtype)
                .to(device=x.device)
                
            )
            self._iSHT_cache[cache_key] = isht
        
        return isht(x)


class SFNOBlock3d(nn.Module):
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
        super(SFNOBlock3d, self).__init__()
        self.spectral_conv = SpectralConv3d(
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

class SFNO3d(nn.Module):
    """
    A 3D Fourier Neural Operator (FNO) model.

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
        super(SFNO3d, self).__init__()
        self.lifting = nn.Conv2d(
            in_channels,
            width,
            kernel_size=1,
        )
        self.fno_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.fno_blocks.append(SFNOBlock3d(
                width,
                width,
                modes,
                activation,
                spectral_hidden_layers,
                spectral_hidden_width,
            ))
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
    


class SpectralConv3d(nn.Module):
    """A 2D Spectral Convolution layer implemented with FFT.
    This layer performs convolution in the Fourier domain by:
    1. Converting input to Fourier space using FFT
    2. Applying a complex-valued neural network to low frequency modes
    3. Converting back to physical space using inverse FFT
    Args:
        in_channels (int): 
            Number of input channels
        out_channels (int): 
            Number of output channels 
        modes (tuple): 
            Number of Fourier modes to preserve in (x,y) directions
        n_layers (int, optional): 
            Number of hidden layers in the complex network. Defaults to 4.
        hidden_width (int, optional): 
            Multiplier for hidden layer width. Defaults to 4.
    Shape:
        - Input: (batch_size, in_channels, spatial_points_x, spatial_points_y)
        - Output: (batch_size, out_channels, spatial_points_x, spatial_points_y)
    Examples:
        >>> layer = SpectralConv2d(in_channels=3, out_channels=1, modes=(12,12))
        >>> x = torch.randn(20, 3, 64, 64)
        >>> output = layer(x)  # shape: [20, 1, 64, 64]
    """

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes, 
            n_layers:int=4, 
            hidden_width=1
        ):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.sht_handle = SHT()
        self.layers = nn.ModuleList([nn.Linear(2*in_channels * modes[0] * modes[1] , hidden_width*out_channels * modes[0] * modes[1])])
        self.layers.extend([nn.Linear(hidden_width*out_channels * modes[0] * modes[1],hidden_width*out_channels * modes[0] * modes[1]) for _ in range(n_layers-2)])
        self.layers.append(nn.Linear(hidden_width*out_channels * modes[0] * modes[1], 2*out_channels * modes[0] * modes[1], bias=False))


    def forward(
            self,
            x
        ):
        batch_size, channels, spatial_points_x, spatial_points_y = x.shape
     
        x_hat = self.sht_handle.sht(x, s=self.modes)
    
        x_hat_under_modes = x_hat[:, :, :self.num_modes_x(spatial_points_x), :self.num_modes_y(spatial_points_y)]

        if x_hat_under_modes.shape[-2] < self.modes[0] or x_hat_under_modes.shape[-1] < self.modes[1]:
            x_hat_under_modes = self.zero_padding_modes(x_hat_under_modes, self.modes[0], self.modes[1])

        out_hat_under_modes = self.network(x_hat_under_modes)

        out_hat = self.zero_padding_fft(out_hat_under_modes, spatial_points_x, spatial_points_y)

        out = self.sht_handle.isht(out_hat, s=(spatial_points_x, spatial_points_y))
        return out.view(batch_size, self.out_channels, spatial_points_x, spatial_points_y)


    def network(
            self, 
            x
        ):
        batch_size, channels, spatial_points_x, spatial_points_y = x.shape
        x = torch.stack((x.real, x.imag),-1)
        x = x.real.reshape(batch_size,-1)#
        #x = 
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.gelu(x)
        x = self.layers[-1](x)
        x = x.reshape(batch_size, channels, spatial_points_x, spatial_points_y,2)
        x = x[:, :, :, :,0] + 1j * x[:, :, :, :,1]
        return x


    def num_modes_x(
            self, 
            spatial_points_x
        ):
        return min(spatial_points_x, self.modes[0])
    

    def num_modes_y(
            self, 
            spatial_points_y
        ):
        return min(spatial_points_y//2 + 1, self.modes[1])
    
    def zero_padding_fft(
            self, 
            x, 
            spatial_points_x, 
            spatial_points_y
        ):
        out_size = (x.size(0), self.out_channels, spatial_points_x, spatial_points_y//2 + 1)
        out = torch.zeros(out_size, dtype=torch.complex64, device=x.device)
        out[:, :, :self.num_modes_x(spatial_points_x), :self.num_modes_y(spatial_points_y)] = x[:, :, :self.num_modes_x(spatial_points_x), :self.num_modes_y(spatial_points_y)]
        return out
    
    def zero_padding_modes(
            self, 
            x, 
            modes_x, 
            modes_y
        ):
        out_size = (x.size(0), self.out_channels, modes_x, modes_y)
        out = torch.zeros(out_size, dtype=torch.complex64, device=x.device)
        out[:, :, :x.shape[-2], :x.shape[-1]] = x
        return out
    
    