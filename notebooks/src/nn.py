import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from pina.model import MultiFeedForward

warnings.filterwarnings("ignore")

from pina import Trainer, Condition, LabelTensor
from pina.problem import AbstractProblem

class Model(torch.nn.Module):
    def __init__(self, input_dimensions, output_dimensions, layers, func):
        super().__init__()

        if not isinstance(input_dimensions, int):
            raise ValueError("input_dimensions expected to be int.")
        self.input_dimension = input_dimensions

        if not isinstance(output_dimensions, int):
            raise ValueError("output_dimensions expected to be int.")
        self.output_dimension = output_dimensions
        
        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1])
            )

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers) - 1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError("Incosistent number of layers and functions")

        unique_list = []
        for layer, func_ in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func_ is not None:
                unique_list.append(func_())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        return self.model(x)
    
    
class Discriminator(nn.Module):
    """
    Autoencoder-style discriminator.
    Encodes inputs to a latent vector, optionally concatenates a conditioning
    representation to the latent, and decodes back to input space.
    forward(x, cond=None) -> (reconstruction, latent)
    """
    def __init__(self, input_dim, latent_dim, hidden_layers=[20], activation=nn.ReLU, cond_dim=0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # build encoder sizes: input -> ... -> latent
        enc_sizes = [input_dim] + hidden_layers + [latent_dim]
        enc_modules = []
        for i in range(len(enc_sizes) - 1):
            enc_modules.append(nn.Linear(enc_sizes[i], enc_sizes[i + 1]))
            if i < len(enc_sizes) - 2:
                enc_modules.append(activation())
        self.encoder = nn.Sequential(*enc_modules)

        # build decoder sizes: (latent + cond) -> ... -> input
        dec_input = latent_dim + cond_dim
        dec_sizes = [dec_input] + list(reversed(hidden_layers)) + [input_dim]
        dec_modules = []
        for i in range(len(dec_sizes) - 1):
            dec_modules.append(nn.Linear(dec_sizes[i], dec_sizes[i + 1]))
            if i < len(dec_sizes) - 2:
                dec_modules.append(activation())
        self.decoder = nn.Sequential(*dec_modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, latent, cond=None):
        # print(cond)
        if self.cond_dim and cond is not None:
            latent = torch.cat([latent, cond], dim=1)
        return self.decoder(latent)

    def forward(self, x):
        """
        x: (B, input_dim)
        returns: reconstruction (B, input_dim), latent (B, latent_dim)
        """
        data = x[0]
        cond = x[1]
        z = self.encode(data)
        recon = self.decode(z, cond)
        return recon