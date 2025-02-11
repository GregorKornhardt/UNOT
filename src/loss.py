"""
loss.py
-------

Function(s) for computing loss values.
"""

import torch


def hilbert_projoection(
        U: torch.Tensor, 
        V: torch.Tensor
    ) -> float:

    """
    Compute the mean Hilbert projective loss between pairs of vectors.

    NB: Usually the Hilbert projective loss takes the logarithm of U and V
    first, but since the output of the predictive network is already in the log
    domain, we omit this step.

    Parameters
    ----------
    U : (n_samples, dim) torch.Tensor
        First set of vectors.
    V : (n_samples, dim) torch.Tensor
        Second set of vectors.

    Returns
    -------
    loss : float
        Mean loss value.
    """

    diff = U - V
    spectrum = torch.max(diff, dim=1)[0] - torch.min(diff, dim=1)[0]
    loss = spectrum.mean()

    return loss