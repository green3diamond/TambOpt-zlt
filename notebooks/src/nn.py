import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from pina.model import FeedForward

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
    
class Generator(nn.Module):

    def __init__(
        self,
        input_dimension=1,
        parameters_dimension=1,
        noise_dimension=1,
        activation=torch.nn.SiLU,
    ):
        super().__init__()

        self._noise_dimension = noise_dimension
        self._activation = activation
        self.model = FeedForward(6 * noise_dimension, input_dimension)
        self.condition = FeedForward(parameters_dimension, 6 * noise_dimension)
        self.parameters_dimension = parameters_dimension

    def forward(self, param):
        # activate to enable noise
        # # uniform sampling in [-1, 1]
        # z = (
        #     2
        #     * torch.rand(
        #         size=(param.shape[0], self._noise_dimension),
        #         device=param.device,
        #         dtype=param.dtype,
        #         requires_grad=True,
        #     )
        #     - 1
        # )
        # return self.model(torch.cat((z, self.condition(param.reshape(-1, self.parameters_dimension))), dim=-1))
        return self.model(self.condition(param.reshape(-1, self.parameters_dimension)))

class Discriminator(nn.Module):

    def __init__(
        self,
        input_dimension=1,
        parameter_dimension=1,
        hidden_dimension=2,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        self._activation = activation
        self.encoding = FeedForward(input_dimension, hidden_dimension)
        self.decoding = FeedForward(2 * hidden_dimension, input_dimension)
        self.condition = FeedForward(parameter_dimension, hidden_dimension)
        self.parameter_dimension = parameter_dimension

    def forward(self, data):
        x, condition = data
        encoding = self.encoding(x)
        conditioning = torch.cat((encoding, self.condition(condition.reshape(-1,self.parameter_dimension))), dim=-1)
        decoding = self.decoding(conditioning)
        return decoding