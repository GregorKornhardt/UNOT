"""
spectral.py
-----------

Implementation of the spectral convolution used in the FNOBlock2d class.
"""
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
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
            hidden_width=4
        ):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes


        self.layers = nn.ModuleList([Complex_Linear_Layer(in_channels, hidden_width*out_channels, modes[0], modes[1])])
        self.layers.extend([Complex_Linear_Layer(hidden_width*out_channels,hidden_width*out_channels, modes[0], modes[1]) for _ in range(n_layers-2)])
        self.layers.append(Complex_Linear_Layer(hidden_width*out_channels, out_channels, modes[0], modes[1], bias=False))


    def forward(
            self,
            x
        ):
        batch_size, channels, spatial_points_x, spatial_points_y = x.shape
        x_hat = torch.fft.rfft2(x)

        x_hat_under_modes = x_hat[:, :, :self.num_modes_x(spatial_points_x), :self.num_modes_y(spatial_points_y)]

        if x_hat_under_modes.shape[-2] < self.modes[0] or x_hat_under_modes.shape[-1] < self.modes[1]:
            x_hat_under_modes = self.zero_padding_modes(x_hat_under_modes, self.modes[0], self.modes[1])

        out_hat_under_modes = self.network(x_hat_under_modes)

        out_hat = self.zero_padding_fft(out_hat_under_modes, spatial_points_x, spatial_points_y)

        out = torch.fft.irfft2(out_hat, s=(spatial_points_x, spatial_points_y))
        return out.view(batch_size, self.out_channels, spatial_points_x, spatial_points_y)


    def network(
            self, 
            x
        ):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.gelu(x.real) + 1j * F.gelu(x.imag)
        x = self.layers[-1](x)
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
    
    
class Complex_Linear_Layer(nn.Module):
    """Complex-valued linear layer for spectral neural networks.
    This layer implements a complex-valued linear transformation in the spectral domain,
    supporting 2D input tensors with multiple channels. It performs complex multiplication
    between the input and weights, with optional complex-valued biases.
    Args:
        in_channels (int): 
            Number of input channels
        out_channels (int): 
            Number of output channels
        modes_x (int): 
            Number of Fourier modes in x direction
        modes_y (int): 
            Number of Fourier modes in y direction
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Shape:
        - Input: (batch_size, in_channels, modes_x, modes_y)
        - Output: (batch_size, out_channels, modes_x, modes_y)
    Attributes:
        real_weights (nn.Parameter): 
            Real part of the weight matrix
        imag_weights (nn.Parameter): 
            Imaginary part of the weight matrix
        real_bias (nn.Parameter or None): 
            Real part of the bias
        imag_bias (nn.Parameter or None): 
            Imaginary part of the bias
    Notes:
        The weights are initialized using Xavier uniform initialization,
        and biases (if used) are initialized to zero.
    """
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            modes_x, 
            modes_y, 
            bias: bool = True
        ):
        super(Complex_Linear_Layer, self).__init__()
        self.in_features = in_channels
        self.out_features = out_channels

        scale = 1.0 / (in_channels * out_channels)

        self.real_weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y))
        self.imag_weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y))
        if bias:
            self.real_bias = nn.Parameter(scale * torch.rand(out_channels, modes_x, modes_y))
            self.imag_bias = nn.Parameter(scale * torch.rand(out_channels, modes_x, modes_y))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)
        
        nn.init.xavier_uniform_(self.real_weights)
        nn.init.xavier_uniform_(self.imag_weights)
        if bias:
            nn.init.zeros_(self.real_bias)
            nn.init.zeros_(self.imag_bias)

    def complex_mult2d(
            self, 
            x_hat, 
            w
        ):
        # Führt elementweise komplexe Multiplikation mit Batch-Unterstützung durch
        real = torch.einsum("b iMN, i o MN -> b o MN", x_hat.real, w.real) - torch.einsum("b iMN, i o MN -> b o MN", x_hat.imag, w.imag)
        imag = torch.einsum("b iMN, i o MN -> b o MN", x_hat.real, w.imag) + torch.einsum("b iMN, i o MN -> b o MN", x_hat.imag, w.real)
        return torch.complex(real, imag)

    def forward(
            self, 
            x
        ):
        weights = torch.complex(self.real_weights, self.imag_weights)

        x = self.complex_mult2d(x, weights) 
        if self.real_bias is not None:
            bias = torch.complex(self.real_bias, self.imag_bias)
            x += bias
        return x