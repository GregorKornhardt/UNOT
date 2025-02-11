import torch
from tqdm import tqdm


def barycenter(
        predictor, 
        MUs, 
        cost_matrix, 
        weights=None, 
        nits=100
    ): 
    """Compute the Wasserstein barycenter of given probability distributions using gradient descent.
    This function implements an iterative algorithm to find the Wasserstein barycenter of multiple
    probability distributions using a neural network predictor and Sinkhorn iterations.
    Args:
        predictor (callable):
            Neural network that predicts optimal transport potentials.
        MUs (torch.Tensor): 
            Input probability distributions of shape (n_distributions, n_points).
        cost_matrix (torch.Tensor): 
            Cost matrix defining the ground metric.
        weights (torch.Tensor, optional): Weights for each input distribution. If None, uniform weights are used.
        nits (int, optional): Number of iterations for the gradient descent. Defaults to 100.
    Returns:
        torch.Tensor: The computed Wasserstein barycenter as a probability distribution.
    Notes:
        - The function uses the Sinkhorn algorithm with a fixed entropic regularization parameter of 0.01
        - The optimization is performed using AdamW optimizer with learning rate 0.1
        - The input distributions and cost matrix should be on the same device (CPU/GPU)
    """
    if weights is None:
        weights = torch.ones(MUs.shape[0], device=MUs.device) / MUs.shape[0]

    v0 = torch.ones(MUs.shape[1],requires_grad = (True), device = MUs.device)
    softmax = torch.nn.Softmax(1)
    opt = torch.optim.AdamW([v0], lr=0.1)
    
    K = torch.exp(-cost_matrix/0.01)
    for k in tqdm(range(nits)):
        
        opt.zero_grad()
        v = predictor(softmax(v0.unsqueeze(0)), softmax(v0.unsqueeze(0)))
        v = torch.exp(v)
        for _ in range(1):
            u = softmax(v0.unsqueeze(0)) / (K @ v.squeeze())
            v = softmax(v0.unsqueeze(0)) / (K.T @ u.squeeze())
        f_mu = torch.log(v) * 0.01

        for i in range(MUs.shape[0]):
            v = predictor(MUs[i].unsqueeze(0), softmax(v0.unsqueeze(0)))
            v = torch.exp(v)
            for _ in range(1):
                u = MUs[i] / (K @ v.squeeze())
                v = softmax(v0.unsqueeze(0)) / (K.T @ u.squeeze())
            #f = torch.log(u) * 0.01
            g = torch.log(v) * 0.01

            if v0.grad is not None:
                v0.grad += weights[i] * (g.flatten() - f_mu.flatten())
            else:
                v0.grad = weights[i] * (g.flatten() - f_mu.flatten())
            
        opt.step()
        
    return softmax(v0.unsqueeze(0))