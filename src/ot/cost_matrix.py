"""
cost_matrix.py
-------

Function(s) for generating cost matrices for optimal transport problems.
"""

import torch
import itertools
import numpy as np

def get_point_cloud(n : int) -> np.ndarray:

    """
    Generate a point cloud representing a 2D grid of n points per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    cloud : (n**2, 2) torch.Tensor
        Point cloud.
    """

    partition = torch.linspace(0, 1, n)
    cloud = torch.stack(torch.meshgrid(partition, partition), axis=-1).reshape(-1, 2)

    return cloud

def get_cost_matrix(
        n : int, 
        device=torch.device("cpu"), 
        dytpe=torch.float32
    ) -> torch.Tensor:
    """
    Generate a square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.
    device : torch.device, optional
        Device on which to create the tensors. Defaults to CPU.
    dytpe : torch.dtype, optional
        Data type of the tensor. Defaults to torch.float32.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """

    cloud = get_point_cloud(n)
    x = np.array(list(itertools.product(cloud, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    C = torch.tensor(np.linalg.norm(a - b, axis=1) ** 2).reshape(n**2, n**2)
    return C.type(dytpe).to(device)

def fast_get_cost_matrix(
        n:int, 
        device=torch.device('cpu')
    ) -> torch.tensor:
    """
    Fast Version for square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.
    device : torch.device, optional
        Device on which to create the tensors. Defaults to CPU.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """
    X = torch.linspace(0,1,n, device=device)
    grid = torch.stack(torch.meshgrid(X,X), dim=2).reshape(-1,2)
    
    norm_X = torch.sum(grid**2, dim=1)
    
    C = (norm_X.unsqueeze(1) @ torch.ones(1, n**2, device=device)) + (torch.ones(n**2, 1, device=device) @ norm_X.unsqueeze(0)) - 2 * grid @ grid.T
    return C 

def fast_get_cost_matrix_XY(
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.tensor:
    """
    Fast Version for square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.
    device : torch.device, optional
        Device on which to create the tensors. Defaults to CPU.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """
    device = X.device 
    n = X.shape[0]
    
    norm_X = torch.sum(X**2, dim=1)
    norm_Y = torch.sum(Y**2, dim=1)
    C = (norm_X.unsqueeze(1) @ torch.ones(1, n, device=device)) + (torch.ones(n, 1, device=device) @ norm_Y.unsqueeze(0)) - 2 * X @ Y.T
    return C 

def get_cost_matrix_l1(n : int, device=torch.device("cpu"), dytpe=torch.float32) -> torch.Tensor:
    """
    Generate a square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """

    cloud = get_point_cloud(n)
    x = np.array(list(itertools.product(cloud, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    C = torch.tensor(np.sum(np.abs(a - b), axis=1)).reshape(n**2, n**2)
    return C.type(dytpe).to(device)

def get_cost_matrix_l2(n : int, device=torch.device("cpu"), dytpe=torch.float32) -> torch.Tensor:
    """
    Generate a square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """

    cloud = get_point_cloud(n)
    x = np.array(list(itertools.product(cloud, repeat=2)))
    a = x[:, 0]
    b = x[:, 1]
    C = torch.tensor(np.linalg.norm(a - b, axis=1)).reshape(n**2, n**2)
    return C.type(dytpe).to(device)

def fast_get_cost_matrix_l2(n:int, device=torch.device('cpu')) -> torch.tensor:
    X = torch.linspace(0,1,n, device=device)
    grid = torch.stack(torch.meshgrid(X,X), dim=2).reshape(-1,2)
    norm_X = torch.sum(grid**2, dim=1)
    C = (norm_X.unsqueeze(1) @ torch.ones(1, n**2, device=device) ) + (torch.ones(n**2, 1, device=device) @ norm_X.unsqueeze(0)) - 2 * grid @ grid.T
    C = torch.sqrt(C)
    C[torch.isnan(C)] = 0
    return C

def get_point_cloud_S2(n : int) -> np.ndarray:

    """
    Generate a point cloud representing a 2D grid of n points per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    cloud : (n**2, 2) torch.Tensor
        Point cloud.
    """

    N_theta = n  # Number of latitude divisions
    N_phi = n    # Number of longitude divisions

    # Define latitude (theta) and longitude (phi)
    theta = torch.linspace(-np.pi/2, np.pi/2, N_theta)
    phi = torch.linspace(0, 2*np.pi, N_phi)

    # Create meshgrid
    Theta, Phi = torch.meshgrid(theta, phi, indexing='ij')

    # Convert to Cartesian coordinates
    R = 1  # Sphere radius
    X = R * torch.cos(Theta) * torch.cos(Phi)
    Y = R * torch.cos(Theta) * torch.sin(Phi)
    Z = R * torch.sin(Theta)

    cloud = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    return cloud

def get_cost_matrix_S2(n : int, device=torch.device("cpu"), dytpe=torch.float32) -> torch.Tensor:
    """
    Generate a square euclidean cost matrix on 2D unit length grid with n points
    per dimension.

    Parameters
    ----------
    n : int
        Number of points per dimension.

    Returns
    -------
    C : (n**2, n**2) torch.Tensor
        Euclidean cost matrix.
    """

    cloud = get_point_cloud_S2(n)
    C = torch.acos(cloud @ cloud.T)
    C[torch.isnan(C)] = 0
    return C.type(dytpe).to(device) / 10