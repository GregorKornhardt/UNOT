"""
geodesics.py
------------
This script contains functions for computing geodesic paths between probability measures.
"""

import torch
import numpy as np
import ot
import math

import src.ot.cost_matrix as cost
from src.evaluation.barycenter import barycenter


def mccann_interpolation_bilinear(P, grid_points, t, N):
    """
    Computes the McCann interpolation at time t using bilinear mass distribution.

        This function calculates the displacement interpolation between two probability measures
        using optimal transport, where the mass is distributed bilinearly to avoid visual artifacts.
        The interpolation is performed on a regular grid in [0,1]^2.

        Parameters
        ----------
        P : ndarray
            (N*N, N*N) array representing the optimal transport plan between source and target measures
        grid_points : ndarray
            (N*N, 2) array containing the coordinates of grid points in [0,1]^2
        t : float
            Interpolation time parameter in [0,1]. t=0 gives source measure, t=1 gives target measure
        N : int
            Grid size (resulting interpolation will be on NxN grid)

        Returns
        -------
        ndarray
            (N, N) array representing the interpolated measure μ_t at time t

        Notes
        -----
        The bilinear distribution helps avoid visual artifacts by spreading mass across
        neighboring grid points based on the precise position of the interpolated points.
        For each mass transfer from point i to j, the interpolated mass is distributed
        to up to 4 neighboring grid points using bilinear weights.
    """
    mu_t = np.zeros((N, N), dtype=float)

    for i in range(N * N):
        mass_from_i = P[i]
        if not np.any(mass_from_i):
            continue

        x_i, y_i = grid_points[i]

        for j in range(N * N):
            m_ij = mass_from_i[j]
            if m_ij == 0:
                continue

            x_j, y_j = grid_points[j]

            # Continuous interpolation
            x_t = (1 - t) * x_i + t * x_j
            y_t = (1 - t) * y_i + t * y_j

            # Map to [0, N-1] space
            X = x_t * (N - 1)
            Y = y_t * (N - 1)

            # integer coords
            iX = int(np.floor(X))
            iY = int(np.floor(Y))

            # fractional offsets
            alpha_x = X - iX
            alpha_y = Y - iY

            # Scatter to up to 4 pixels
            for dy in [0, 1]:
                for dx in [0, 1]:
                    rr = iY + dy
                    cc = iX + dx
                    if 0 <= rr < N and 0 <= cc < N:
                        weight = (alpha_x if dx == 1 else (1 - alpha_x)) * (
                            alpha_y if dy == 1 else (1 - alpha_y)
                        )
                        mu_t[rr, cc] += m_ij * weight

    return mu_t


def geodesic_interpolation(X, t):
    pi_t = (1 - t) * X[:, None, :] + t * X[None, :, :]
    return pi_t.reshape(-1, 2)


def mccann_network(
    predictor,
    mu,
    nu,
):
    """
    Compute the McCann interpolation between two discrete measures using a neural network.
    This function uses a predictor network to initialize the dual variables for the Sinkhorn algorithm,
    then computes the optimal transport plan and generates interpolated measures along the geodesic.
    Parameters
    ----------
    predictor : torch.nn.Module
        Neural network that predicts the dual variable f given input measures mu and nu
    mu : torch.Tensor
        Source measure
    nu : torch.Tensor
        Target measure
    device : str, optional
        Device to perform computations on ('cpu' or 'cuda'), by default 'cpu'
    Returns
    -------
    list
        List of interpolated measures along the geodesic path.
        Contains 5 measures corresponding to t = 0, 0.25, 0.5, 0.75, 1.
    Notes
    -----
    The function uses the Sinkhorn algorithm with a fixed entropic regularization parameter of 0.01
    and generates interpolations at 5 equally spaced time points between 0 and 1.
    """
    f = predictor(mu, nu)
    v = torch.exp(f).flatten()

    length = math.isqrt(mu.shape[0])
    device = mu.device

    X = torch.tensor(cost.get_point_cloud(length), dtype=torch.float32)
    cost_matrix = cost.get_cost_matrix(length, device=device)

    K = torch.exp(-cost_matrix.to(device) / 0.01)

    u = mu / (K @ v)
    v = nu / (K.T @ u)
    G = torch.diag(u) @ K @ torch.diag(v)

    plan_geodesic = []
    for i in range(0, 5):
        t = i / 4
        gamma_t = mccann_interpolation_bilinear(
            G.cpu().detach().numpy(), X.cpu().detach().numpy(), t, 28
        )
        plan_geodesic.append(gamma_t)

    return plan_geodesic


def barycenter_geodesic(predictor, mu, nu, cost_matrix, device="cpu"):
    """
    Compute barycentric geodesic between two probability distributions.
    This function calculates intermediate points along the geodesic path between two
    probability distributions using Wasserstein barycenters. It computes 5 points along
    the path with weights [1,0], [0.75,0.25], [0.5,0.5], [0.25,0.75], [0,1].
    Parameters
    ----------
    predictor : torch.nn.Module
        Neural network model that predicts transport maps
    mu : torch.Tensor
        Source probability distribution
    nu : torch.Tensor
        Target probability distribution
    cost_matrix : torch.Tensor
        Matrix of transportation costs between points
    device : str, optional
        Device to perform computations on ('cpu' or 'cuda'), by default 'cpu'
    Returns
    -------
    list
        List of numpy arrays containing the transport maps at different points
        along the geodesic path between mu and nu
    Notes
    -----
    The function uses the barycenter algorithm with 200 iterations for each point
    along the geodesic path.
    """
    bar = torch.stack((mu, nu))
    bar_geodesic = []
    weights_list = [[1, 0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0, 1]]

    for weights in weights_list:
        gamma_t = barycenter(
            predictor,
            bar,
            cost_matrix.to(device),
            torch.tensor(weights).to(device),
            nits=200,
        )
        bar_geodesic.append(gamma_t.cpu().detach().numpy())

    return bar_geodesic


def true_mccann_geodesic(
    mu,
    nu,
):
    """
    Computes the true McCann geodesic between two probability distributions using optimal transport.
    The function calculates the optimal transport plan between the input distributions and
    performs McCann interpolation to generate points along the geodesic path.
    Parameters:
        mu (torch.Tensor):
            Source probability measure
        nu (torch.Tensor):
            Target probability measure
    Returns:
        list: A list containing 5 equally spaced points along the geodesic path from mu to nu,
              including the endpoints. Each point is a probability distribution obtained through
              McCann interpolation.
    Notes:
        - Uses Sinkhorn algorithm with entropic regularization (reg=0.01) for computing optimal transport
        - Generates 5 interpolation points at t = 0, 0.25, 0.5, 0.75, 1
        - Input distributions must be compatible with the predefined cost matrix dimensions
    """

    length = math.isqrt(mu.shape[0])
    cost_matrix = cost.get_cost_matrix(length)

    X = torch.tensor(cost.get_point_cloud(length), dtype=torch.float32)

    G = ot.bregman.sinkhorn(
        mu.cpu().detach().numpy(),
        nu.cpu().detach().numpy(),
        cost_matrix.cpu().numpy(),
        reg=0.01,
    )
    true_geodesic = []

    for i in range(0, 5):
        t = i / 4
        gamma_t = mccann_interpolation_bilinear(G, X.cpu().detach().numpy(), t, 28)
        true_geodesic.append(gamma_t)

    return true_geodesic
