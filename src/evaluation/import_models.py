import torch
import math
from typing import Tuple
from ..networks.FNO2d import FNO2d
from ..networks.mlp import Predictor, Predictor_Var_Eps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mlp(
    name: str,
    dimension: int = 28,
    width_predictor: int = 4,
    num_layers: int = 3,
    device: str = device,
) -> torch.nn.Module:
    """
    Load a Multi-Layer Perceptron (MLP) model for use in predictions.

    This function loads a pre-trained MLP model from a specified file and returns it for use in inference tasks.
    The model's architecture can be customized with the `dimension`, `width_predictor`, and `num_layers`
    parameters, and it can be loaded onto a specified device (CPU or GPU).

    Parameters
    ----------
    name : str
        The name of the model file (without extension) to load. The model should be located in the 'Models/' directory.
    dimension : int, optional
        The input dimension of the model. Default is 28, but typically corresponds to the size of input data (e.g., 28x28 pixels for images).
    width_predictor : int, optional
        A scaling factor for the width of the hidden layers. It adjusts the number of units in each layer based on `dimension`. Default is 4.
    num_layers : int, optional
        The number of hidden layers in the MLP. Default is 3.
    device : str, optional
        The device to load the model onto. Can be either 'cpu' or 'cuda'.

    Returns
    -------
    predictor : torch.nn.Module
        The loaded MLP model, ready for inference.

    Example
    -------
    >>> # Load model and move it to GPU
    >>> predictor = load_mlp('model_name', dimension=28**2, device='cuda')
    >>> mu = ...  # Shape: (batch_size, dimension)
    >>> nu = ...  # Shape: (batch_size, dimension)
    >>> g = torch.exp(predictor(mu, nu))  # Get prediction using the model

    Notes
    -----
    - The model file should be saved in the 'Models/' directory with the extension `.pt`.
    - The MLP model is defined with the input dimension, width of layers, and number of layers provided during loading.
    - Make sure the `dimension` matches the input data shape (e.g., if working with flattened image data, it could be 28*28 for MNIST images).
    """

    predictor = Predictor(
        dimension**2, int(width_predictor * dimension**2), num_layers
    ).to(device)
    predictor.load_state_dict(torch.load("Models/" + name + ".pt", map_location=device))
    predictor.eval()

    return predictor


def apply_fno(predictor, mu, nu, grid=False):
    """
    Apply the Fourier Neural Operator (FNO) model to two input tensors.

    This function takes two input tensors `mu` and `nu`, reshapes them, and passes them through
    the predictor (FNO model) to obtain the output. The inputs should typically represent two
    fields (e.g., solutions of partial differential equations) that will be processed by the
    Fourier neural operator.

    Parameters
    ----------
    predictor : torch.nn.Module
        The pre-trained Fourier Neural Operator (FNO) model used for prediction.
    mu : torch.Tensor
        The first input tensor of shape (batch_size, dimension). Typically represents the first field.
    nu : torch.Tensor
        The second input tensor of shape (batch_size, dimension). Typically represents the second field.
    grid : bool
        If True, the spatial grid is included as an additional input to the model. If False, the grid is not included. Default is False.

    Returns
    -------
    torch.Tensor
        The output tensor after applying the FNO, reshaped to match the expected output dimensions.

    Example
    -------
    >>> # Assuming 'predictor' is already loaded FNO model
    >>> mu = ...  # Shape: (batch_size, dimension)
    >>> nu = ...  # Shape: (batch_size, dimension)
    >>> output = apply_fno(predictor, mu, nu)  # Apply the model to inputs

    Notes
    -----
    - This function assumes that the input tensors `mu` and `nu` are flattened into 1D vectors before being processed.
    - The model expects the input shape to be reshaped into (batch_size, 2, length, length), where `length` is the square root of the input dimension.
    """
    length = int(math.isqrt(mu.shape[-1]))  # Compute the side length of the square grid

    if grid:
        grid = get_grid(length)
        input = torch.cat(
            (
                mu.reshape(-1, 1, length, length),
                nu.reshape(-1, 1, length, length),
                grid.repeat(mu.shape[0], 1, 1, 1).to(nu.device),
            ),
            1,
        )
    else:
        input = torch.cat(
            (mu.reshape(-1, 1, length, length), nu.reshape(-1, 1, length, length)), 1
        )

    return predictor(input).reshape(-1, length * length)


def load_fno(
    name: str,
    modes: Tuple[int, int] = (10, 10),
    width: int = 64,
    activation: torch.nn.Module = torch.nn.GELU(),
    grid: bool = False,
    device: str = device,
) -> torch.nn.Module:
    """
    Load a pre-trained Fourier Neural Operator (FNO) model for inference.

    This function loads a pre-trained FNO model. The model's architecture, including the number of modes
    and width, can be configured through the function parameters.

    Parameters
    ----------
    name : str
        The name of the model file (without extension) to load. The model should be located in the 'Models/' directory.
    modes : tuple of int, optional
        The number of Fourier modes used in the model. Default is (10, 10), which determines the frequency resolution.
    width : int, optional
        The width (number of channels) of the hidden layers in the network. Default is 64.
    activation : torch.nn.Module, optional
        The activation function to be used in the model. Default is GELU.
    grid : bool, optional
        If True, the spatial grid is included as an additional input to the model. If False, the grid is not included. Default is False.
    device : str, optional
        The device to load the model onto. Can be either 'cpu', 'cuda', or 'mps'.

    Returns
    -------
    predictor : callable
        A lambda function that takes two inputs (`mu` and `nu`) and applies the FNO model to them.

    Example
    -------
    >>> # Load model and move it to GPU
    >>> predictor = load_fno('model_name', in_channels=2, modes=(10, 10), width=64, device='cuda')
    >>> mu = ...  # Shape: (batch_size, dimension)
    >>> nu = ...  # Shape: (batch_size, dimension)
    >>> output = torch.exp(predictor(mu, nu))  # Get prediction using the model

    Notes
    -----
    - The model file should be saved in the 'Models/' directory with the extension `.pt`.
    - This implementation assumes that the input tensors `mu` and `nu` represent fields with a square grid structure.
    """

    # Initialize the FNO model with the given parameters
    in_channels = 4 if grid else 2

    predictor = FNO2d(in_channels, 1, modes, width, activation=activation).to(device)

    # Load the pre-trained model weights from file
    predictor.load_state_dict(
        torch.load("Models/" + name + ".pt", map_location=device, weights_only=True)
    )

    # Set the model to evaluation mode for inference
    predictor.eval()

    # Return a function to apply the FNO model
    return lambda mu, nu: apply_fno(predictor, mu, nu, grid)


def get_grid(length):
    """
    Generate a 2D grid of shape (2, length, length) representing coordinates in a unit square.

    This function generates a 2D grid where each point corresponds to a (x, y) coordinate in the unit square [0, 1] x [0, 1].
    The grid is commonly used as a spatial reference for fields defined on a 2D domain.

    Parameters
    ----------
    length : int
        The length of each side of the square grid. The grid will have shape (2, length, length), where the first dimension
        represents the x and y coordinates respectively.

    Returns
    -------
    torch.Tensor
        A tensor of shape (2, length, length) containing the x and y coordinates of the grid.

    Example
    -------
    >>> grid = get_grid(10)  # Generate a 10x10 grid of (x, y) coordinates
    """
    x = torch.linspace(0, 1, length)  # Generate x coordinates
    y = torch.linspace(0, 1, length)  # Generate y coordinates
    grid = torch.stack(torch.meshgrid(x, y))  # Stack them into a 2D grid (x, y)
    return grid


def apply_predictor(predictor, mu, nu, eps, grid):
    """
    Apply a model to two input fields (mu, nu) along with optional grid and epsilon tensors.

    This function takes two input tensors `mu` and `nu`, reshapes them, and optionally includes a 2D spatial grid and
    an epsilon tensor (typically used for regularization or boundary conditions). The reshaped inputs are then passed through
    the predictor to obtain the output. The grid is only included if specified.

    Parameters
    ----------
    predictor : torch.nn.Module
        The pre-trained model used for predictions. This is typically a Fourier Neural Operator (FNO).
    mu : torch.Tensor
        The first input field, of shape (batch_size, dimension).
    nu : torch.Tensor
        The second input field, of shape (batch_size, dimension).
    eps : torch.Tensor
        The epsilon tensor, used for regularization or boundary conditions. Should have the same shape as `mu`.
    grid : bool
        If True, the spatial grid is included as an additional input to the model. If False, the grid is not included.

    Returns
    -------
    torch.Tensor
        The output tensor after applying the model to the inputs. The output is reshaped to match the expected output dimensions.

    Example
    -------
    >>> # Assuming 'predictor' is already loaded model
    >>> mu = ...  # Shape: (batch_size, dimension)
    >>> nu = ...  # Shape: (batch_size, dimension)
    >>> eps = ... # Shape: (batch_size, dimension)
    >>> output = apply_predictor(predictor, mu, nu, eps, grid=True)  # Apply the model to inputs with grid

    Notes
    -----
    - If `grid` is True, the grid is generated using the `get_grid` function and is added to the input.
    - The `mu`, `nu`, and `eps` tensors are reshaped to match the model's input dimensions, with a total of 3 or 5 channels (depending on the inclusion of the grid).
    """
    length = int(math.isqrt(mu.shape[-1]))  # Compute the side length of the square grid

    if grid:
        grid = get_grid(length)  # Generate the spatial grid
        # Concatenate mu, nu, grid, and eps into the input tensor
        input = torch.cat(
            (
                mu.reshape(-1, 1, length, length),
                nu.reshape(-1, 1, length, length),
                grid.repeat(mu.shape[0], 1, 1, 1).to(nu.device),
                eps * torch.ones_like(mu).reshape(-1, 1, length, length),
            ),
            1,
        )
    else:
        input = torch.cat(
            (
                mu.reshape(-1, 1, length, length),
                nu.reshape(-1, 1, length, length),
                eps * torch.ones_like(mu).reshape(-1, 1, length, length),
            ),
            1,
        )

    return predictor(input.float()).reshape(-1, length * length)


def load_fno_var_epsilon(
    name: str,
    modes: Tuple[int, int] = (14, 14),
    width: int = 128,
    activation: torch.nn.Module = torch.nn.GELU(),
    grid: bool = True,
    fixed_eps: bool = False,
    device: str = "mps",
) -> torch.nn.Module:
    """
    Load the Fourier Neural Operator (FNO) model for use with variable epsilon values.

    This function loads a pre-trained FNO model, which can take two input fields (`mu` and `nu`) along with an epsilon tensor
    for regularization or boundary conditions. The model is designed to handle variable epsilon values during inference.

    Parameters
    ----------
    name : str
        The name of the model file (without extension) to load. The model should be located in the 'Models/' directory.
    modes : tuple of int, optional
        The number of Fourier modes used in the model. Default is (14, 14), which determines the frequency resolution.
    width : int, optional
        The width (number of channels) of the hidden layers in the network. Default is 128.
    activation : torch.nn.Module, optional
        The activation function to be used in the model. Default is GELU.
    grid : bool, optional
        If True, the spatial grid is included as an additional input to the model. If False, the grid is not included. Default is True.
    fixed_eps : bool, optional
        If True, the epsilon value is fixed to 1e-2 and does not need to be provided during inference. Default is False.
    device : str, optional
        The device to load the model onto. Can be either 'cpu', 'cuda', or 'mps' for Apple M1 chips. Default is 'mps'.

    Returns
    -------
    predictor : torch.nn.Module
        The loaded model, ready for inference with variable epsilon inputs.

    Example
    -------
    >>> # Load model and move it to GPU
    >>> predictor = load_fno_var_epsilon('model_name', in_channels=2, modes=(14, 14), width=128, device='cuda')
    >>> mu = ...  # Shape: (batch_size, dimension)
    >>> nu = ...  # Shape: (batch_size, dimension)
    >>> eps = ... # Shape: (batch_size, dimension)
    >>> output = apply_predictor(predictor, mu, nu, eps, grid=True)  # Get prediction with epsilon regularization

    Notes
    -----
    - The model file should be saved in the 'Models/' directory with the extension `.pt`.
    - This implementation assumes that the input tensors `mu` and `nu` represent fields with a square grid structure.
    """

    # Initialize the FNO model with the given parameters
    in_channels = 5 if grid else 3
    predictor = FNO2d(in_channels, 1, modes, width, activation=activation).to(device)

    # Load the pre-trained model weights from file
    predictor.load_state_dict(
        torch.load("Models/" + name + ".pt", map_location=device, weights_only=True)
    )

    # Set the model to evaluation mode for inference
    predictor.eval()

    if fixed_eps:
        eps = torch.tensor(1e-2).to(device)
        return lambda mu, nu: apply_predictor(predictor, mu, nu, eps, grid)

    return lambda mu, nu, eps: apply_predictor(predictor, mu, nu, eps, grid)
