import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

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