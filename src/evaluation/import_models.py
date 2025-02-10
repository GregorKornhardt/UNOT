import torch
import src.networks.FNO2d as FNO
#from src.networks.neuralop.models.fno import FNO2d


def load_model(name : str,
               dimension : int = 28**2,
               device : str = 'mps') -> torch.nn.Module:
    '''
    Load the model.
    
    Parameters
    ----------
    name : str
        Name of the model to use.
    dimension : int
        Dimension of the model.
    device : str
        Device to use.

    Returns
    -------
    predictor : torch.nn.Module
        The loaded model.
    '''
    if dimension == 28**2:
        width_predictor = 4 * dimension
    if dimension == 64**2:
        width_predictor = 2 * dimension

    #predictor = networks.Predictor(dimension, width_predictor, 3).to(device)
    predictor = Predictor(dimension, width_predictor).to(device)
    predictor.load_state_dict(torch.load('Models/' + name + '.pt', map_location=device))
    predictor.eval()
    
    return predictor


def load_fno(
        name : str, 
        device : str = 'mps'
    ) -> torch.nn.Module:
    '''
    Load the model.
    
    Parameters
    ----------
    name : str
        Name of the model to use.
    dimension : int
        Dimension of the model.
    device : str
        Device to use.

    Returns
    -------
    predictor : torch.nn.Module
        The loaded model.
    '''
    

    predictor = FNO.FNO2d(2, 1, (10,10), 64, activation=torch.nn.GELU()).to(device)
    """predictor = FNO2d(
                n_modes_height=10, 
                n_modes_width=10,
                in_channels=2,
                hidden_channels=64,
                activation=torch.nn.GELU(), 
                n_layers=4).to(device)"""
    predictor.load_state_dict(torch.load('Models/' + name + '.pt', map_location=device, weights_only=True))
    predictor.eval()
    
    return predictor



def load_fno_var_epsilon(name : str,
               device : str = 'mps') -> torch.nn.Module:
    '''
    Load the model.
    
    Parameters
    ----------
    name : str
        Name of the model to use.
    dimension : int
        Dimension of the model.
    device : str
        Device to use.

    Returns
    -------
    predictor : torch.nn.Module
        The loaded model.
    '''
    #predictor = FNO.FNO2d(3, 1, (10,10), 64, activation=torch.nn.GELU()).to(device)

    predictor = FNO.FNO2d(5, 1, (14,14), 128, activation=torch.nn.GELU()).to(device)

    predictor.load_state_dict(torch.load('Models/' + name + '.pt', map_location=device, weights_only=True))
    predictor.eval()
    
    return predictor

import torch.nn as nn
class Predictor(nn.Module):

    """
    Predictive network class.
    """

    def __init__(self, dim : int, width : int):

        """
        Parameters
        ----------
        dim : int
            Dimension of each probability distribution in the data.
        width : int
            Width of the predictive network hidden layers.
        """

        super(Predictor, self).__init__()
        self.dim = dim
        self.width = width
        self.l_1 = nn.Sequential(nn.Linear(2*dim, width), nn.BatchNorm1d(width), nn.ELU())
        self.l_2 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ELU())
        #self.l_3 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ELU())
        #self.l_4 = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ELU())
        self.l_3 = nn.Sequential(nn.Linear(width, dim))
        #self.layers = [self.l_1, self.l_2, self.l_3, self.l_4, self.l_5]  
        self.layers = [self.l_1, self.l_2, self.l_3]
     

    def forward(self, x_a, x_b):
        x = torch.cat((x_a, x_b), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
