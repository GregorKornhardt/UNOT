"""
sinkhorn.py
-----------

Implementation(s) of the Sinkhorn algorithm and associated functions for
computing solutions to the entropic regularized optimal transport problem.
"""

# Imports
import torch
from typing import Tuple


def sink(
    mu: torch.Tensor,
    nu: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    v0: torch.Tensor,
    maxiter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    The standard Sinkhorn algorithm!

    Parameters
    ----------
    mu : (dim,) torch.Tensor
        First probability distribution.
    nu : (dim,) torch.Tensor
        Second probability distribution.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    v0 : (dim,) torch.Tensor
        Initial guess for scaling factor v.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    u : (dim,) torch.Tensor
        1st Scaling factor.
    v : (dim,) torch.Tensor
        2nd Scaling factor.
    G : (dim, dim) torch.Tensor
        Optimal transport plan.
    dist : float
        Optimal transport distance.
    """

    K = torch.exp(-C / eps)
    v = v0

    for _ in range(maxiter):
        u = mu / (K @ v)
        v = nu / (K.T @ u)

    G = torch.diag(u) @ K @ torch.diag(v)
    dist = torch.trace(C.T @ G)

    return u, v, G, dist


def sink_vec(
    MU: torch.Tensor,
    NU: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    V0: torch.Tensor,
    n_iters: int,
) -> torch.Tensor:
    """
    A vectorized version of the Sinkhorn algorithm to create scaling factors
    to be used for generating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    V0 : (n_samples, dim) torch.Tensor
        Initial guess for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factor.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factor.
    """

    K = torch.exp(-C / eps)
    V = V0

    for _ in range(n_iters):
        U = MU / (K @ V.T).T
        V = NU / (K.T @ U.T).T

    return U, V


def sink_vec_dist(
    MU: torch.Tensor,
    NU: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    V0: torch.Tensor,
    n_iters: int,
) -> torch.Tensor:
    """
    A vectorized version of the Sinkhorn algorithm to create scaling factors
    to be used for generating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    V0 : (n_samples, dim) torch.Tensor
        Initial guess for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factor.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factor.
    """
    K = torch.exp(-C / eps)
    V = V0

    for _ in range(n_iters):
        U = MU / (K @ V.T).T
        V = NU / (K.T @ U.T).T

    D = K * C  # Größe: [i, j]
    dist = torch.einsum("si,ij,sj->s", U, D, V)  # Größe: [s]
    return U, V, dist


# @torch.jit.script
def sink_vec_eps(
    MU: torch.Tensor,
    NU: torch.Tensor,
    C: torch.Tensor,
    eps: torch.Tensor,
    V0: torch.Tensor,
    n_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A vectorized version of the Sinkhorn algorithm to create scaling factors
    to be used for generating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        epsilonularization parameter.
    V0 : (n_samples, dim) torch.Tensor
        Initial guess for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factor.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factor.
    """
    C = C.repeat(eps.shape[0], 1, 1)
    eps = eps.unsqueeze(1).unsqueeze(1)
    K = torch.exp(-C / eps)

    V = V0.unsqueeze(2)
    U = V.clone()
    MU = MU.unsqueeze(2)
    NU = NU.unsqueeze(2)

    for _ in range(n_iters):
        U = MU / torch.bmm(K, V)
        V = NU / torch.bmm(torch.transpose(K, 1, 2), U)

    return U.squeeze(2), V.squeeze(2)
