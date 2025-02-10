"""
mlp.py
-------

Pytorch neural network classes for the predictive MLPs.
"""
import torch
import torch.nn as nn

class Predictor_Var_Eps(nn.Module):

    """
    Predictive network class.
    """

    def __init__(self, dim : int, width : int, n_layers : int = 3):

        """
        Parameters
        ----------
        dim : int
            Dimension of each probability distribution in the data.
        dim_hidden : int
            Dimension of the hidden layers.
        num_layers : int
            Number of hidden layers.
        """

        super(Predictor_Var_Eps, self).__init__()
        self.dim = dim
        self.width = width

        # Create a list to store the layers of the network
        self.layers = torch.nn.ModuleList()
        # Add the first layer
        self.layers.append(nn.Sequential(nn.Linear(2*dim + 1, width), nn.BatchNorm1d(width), nn.ELU()))
        # Add the hidden layers
        for i in range(n_layers-1):
            self.layers.append(nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ELU()))
        # Add the final layer
        self.layers.append(nn.Sequential(nn.Linear(width, dim)))
        
        # initialize the layers using glorot initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
     

    def forward(self, x_a, x_b, epsilon):
        x = torch.cat((x_a, x_b, epsilon), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class Predictor(nn.Module):

    """
    Predictive network class.
    """

    def __init__(self, dim : int, dim_hidden : int, num_layers : int):

        """
        Parameters
        ----------
        dim : int
            Dimension of each probability distribution in the data.
        dim_hidden : int
            Dimension of the hidden layers.
        num_layers : int
            Number of hidden layers.
        """

        super(Predictor, self).__init__()

        self.layers = torch.nn.ModuleList()

        self.layers.append(nn.Sequential(nn.Linear(2*dim, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))
        self.layers.append(nn.Sequential(nn.Linear(dim_hidden, dim)))
     

    def forward(self, x_a, x_b):
        x = torch.cat((x_a, x_b), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x