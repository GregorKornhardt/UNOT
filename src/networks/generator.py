"""
generator.py
-------

Pytorch neural network classes for the generative MLPs.
"""

import torch
import torch.nn as nn
import torchvision


class Generator(nn.Module):
    """
    Measure generating network class.
    """

    def __init__(self, dim_prior : int, dim : int, dim_hidden : int, num_layers : int,
                 dust_const : float, skip_const : float):

        """
        Parameters
        ----------
        dim_prior : int
            Dimension of the prior distribution.
        dim : int
            Dimension of each probability distribution in the data.
        dim_hidden : int
            Dimension of the hidden layers.
        num_layers : int
            Number of hidden layers.
        dust_const : float
            Constant to add to the images to avoid zero entries.
        skip_const : float
            Constant to control the strength of the skip connection.
        """

        super(Generator, self).__init__()
        self.dim_prior = dim_prior
        self.dim = dim
        self.dust_const = dust_const
        self.skip_const = skip_const
        self.length_prior = int(self.dim_prior**.5)
        self.length = int(self.dim**.5)
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(nn.Sequential(nn.Linear(2*dim_prior, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))
        
        # Hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))

        # Output layer
        self.layers.append(nn.Sequential(nn.Linear(dim_hidden, 2*dim), nn.Sigmoid()))


    def forward(self, x):

        # Creating a reshaped copy of the input to use as a skip connection
        x_0 = x.reshape(2, x.size(0), self.length_prior, self.length_prior)
        transform = torchvision.transforms.Resize((self.length, self.length),antialias=True)
        x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim), transform(x_0[1]).reshape(x.size(0), self.dim)), 1)

        # Forward pass
        for layer in self.layers:
            x = layer(x)

        x = x + self.skip_const * x_0
        x = nn.functional.relu(x)

        x_a = x[:, :self.dim]
        x_b = x[:, self.dim:]
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x_a = x_a + self.dust_const
        x_b = x_b + self.dust_const
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        #x = torch.cat((x_a, x_b), dim=1)

        return x_a, x_b

class Generator_Var_Eps(nn.Module):
    """
    Measure generating network class with changing measures over the regularization parameter epsilion.
    """

    def __init__(self, dim_prior : int, dim : int, dim_hidden : int, num_layers : int,
                 dust_const : float, skip_const : float):

        """
        Parameters
        ----------
        dim_prior : int
            Dimension of the prior distribution.
        dim : int
            Dimension of each probability distribution in the data.
        dim_hidden : int
            Dimension of the hidden layers.
        num_layers : int
            Number of hidden layers.
        dust_const : float
            Constant to add to the images to avoid zero entries.
        skip_const : float
            Constant to control the strength of the skip connection.
        """

        super(Generator_Var_Eps, self).__init__()
        self.dim_prior = dim_prior
        self.dim = dim
        self.dust_const = dust_const
        self.skip_const = skip_const
        self.length_prior = int(self.dim_prior**.5)
        self.length = int(self.dim**.5)
        self.layers = torch.nn.ModuleList()

        # Add the first layer
        self.layers.append(nn.Sequential(nn.Linear(2*dim_prior + 1, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))
        
        # Add the hidden layers
        for i in range(num_layers-1):
            self.layers.append(nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ELU()))
        
        # Add the final layer
        self.layers.append(nn.Sequential(nn.Linear(dim_hidden, 2*dim), nn.Sigmoid()))

        # initialize the layers using glorot initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        

    def forward(self, x, epsilon):
        # Creating a reshaped copy of the input to use as a skip connection
        x_0 = x.reshape(2, x.size(0), self.length_prior, self.length_prior)
        transform = torchvision.transforms.Resize((self.length, self.length),antialias=True)
        x_0 = torch.cat((transform(x_0[0]).reshape(x.size(0), self.dim), transform(x_0[1]).reshape(x.size(0), self.dim)), 1)

        x = torch.cat((x, epsilon), 1)
        # Forward pass
        for layer in self.layers:
            x = layer(x)

        x = x + self.skip_const * x_0
        x = nn.functional.relu(x)

        x_a = x[:, :self.dim]
        x_b = x[:, self.dim:]
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        x_a = x_a + self.dust_const
        x_b = x_b + self.dust_const
        x_a = x_a / torch.unsqueeze(x_a.sum(dim=1), 1)
        x_b = x_b / torch.unsqueeze(x_b.sum(dim=1), 1)
        #x = torch.cat((x_a, x_b), dim=1)

        return x_a, x_b