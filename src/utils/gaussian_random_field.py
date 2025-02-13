"""
gaussian_random_field.py
------------------------

Generator for 2D scale-invariant Gaussian Random Fields using PyTorch.
Adapted from https://github.com/bsciolla/gaussian-random-fields
"""

# Main dependencies
import torch
import numpy as np


def fftind(size):
    """Returns a torch tensor of shifted Fourier coordinates k_x k_y.

    Input args:
        size (integer): The size of the coordinate array to create
    Returns:
        k_ind, torch tensor of shape (2, size, size) with:
            k_ind[0,:,:]:  k_x components
            k_ind[1,:,:]:  k_y components

    Example:

        print(fftind(5))

        [[[ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]
        [ 0  1 -3 -2 -1]]

        [[ 0  0  0  0  0]
        [ 1  1  1  1  1]
        [-3 -3 -3 -3 -3]
        [-2 -2 -2 -2 -2]
        [-1 -1 -1 -1 -1]]]

    """
    k_ind = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size))) - int(
        (size + 1) / 2
    )
    k_ind = torch.fft.fftshift(k_ind)
    return k_ind


def gaussian_random_field(alpha=3.0, size=128, flag_normalize=True):
    """Returns a torch tensor of shifted Fourier coordinates k_x k_y.

    Input args:
        alpha (double, default = 3.0):
            The power of the power-law momentum distribution
        size (integer, default = 128):
            The size of the square output Gaussian Random Fields
        flag_normalize (boolean, default = True):
            Normalizes the Gaussian Field:
                - to have an average of 0.0
                - to have a standard deviation of 1.0

    Returns:
        gfield (torch tensor of shape (size, size)):
            The random gaussian random field

    Example:
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example)
    """

    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = torch.pow(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise_real = torch.randn(size, size)
    noise_imag = torch.randn(size, size)
    noise = noise_real + 1j * noise_imag

    # To real space
    gfield = torch.fft.ifft2(noise * amplitude).real

    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - torch.mean(gfield)
        gfield = gfield / torch.std(gfield)

    return gfield


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for i in range(20):
        plt.figure()
        plt.title(f"Example {i/2}")
        example = gaussian_random_field(alpha=i / 2, size=64)
        plt.subplot(1, 2, 1)
        plt.title(f"Example {i/2} Squared")
        plt.imshow(example**2, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"Example {i/2} Non-Squared")
        plt.imshow(example, cmap="gray")
    plt.show()
