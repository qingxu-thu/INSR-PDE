# --- bulit in ---
import math
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
from typing import Any, List, Dict, Union, Tuple, Optional, Callable
from .diff_ops import *
#from .gridencoder import GridEncoder
import tinycudann as tcnn

# --- my module ---

"""
The MIT License (MIT)
Copyright (c) 2022 Joe Hsiao (Ending2015a)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

# --- constants ---
PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]


class Frequency(nn.Module):
    def __init__(
        self,
        dim: int,
        n_levels: int = 10
    ):
        """Positional encoding from NeRF: https://www.matthewtancik.com/nerf
        [sin(x), cos(x), sin(4x), cos(4x), sin(8x), cos(8x),
        ..., sin(2^n*x), cos(2^n*x)]

        Args:
        dim (int): input dimensions
        n_levels (int, optional): number of frequencies. Defaults to 10.
        """
        super().__init__()
        self.n_levels = n_levels
        assert self.n_levels > 0
        freqs = 2. ** torch.linspace(0., n_levels-1, n_levels)
        self.register_buffer('freqs', freqs, persistent=False)
        # ---
        self.input_dim = dim
        self.output_dim = dim * n_levels * 2
    
    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=-1) # (..., dim, 1)
        x = x * self.freqs # (..., dim, L)
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1) # (..., dim, L*2)
        return x.flatten(-2, -1) # (..., dim * L * 2)


class Simple_MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int = 1,
        mlp_units: List[int] = []
    ):
        super().__init__()
        layers = []
        self.input_dim = dim
        self.output_dim = out_dim
        in_dim = dim
        for out_dim in mlp_units:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.tensor):
        return self.model(x)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=np.sqrt(1.5505188080679277) / np.sqrt(num_input))


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.extend([nn.Linear(in_features, hidden_features), nl])

        for i in range(num_hidden_layers):
            self.net.extend([nn.Linear(hidden_features, hidden_features), nl])

        self.net.append(nn.Linear(hidden_features, out_features))
        if not outermost_linear:
            self.net.append(nl)

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, weights=None):
        output = self.net(coords)
        if weights is not None:
            output = output * weights
        return output


class MultiResHashGrid(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int, 
        cfg
    ):
        """NVidia's hash grid encoding
        https://nvlabs.github.io/instant-ngp/

        The output dimensions is `n_levels` * `n_features_per_level`,
        or your can simply access `model.output_dim` to get the output dimensions

        Args:
        dim (int): input dimensions, supports at most 7D data.
        n_levels (int, optional): number of grid levels. Defaults to 16.
        n_features_per_level (int, optional): number of features per grid level.
            Defaults to 2.
        log2_hashmap_size (int, optional): maximum size of the hashmap of each
            level in log2 scale. According to the paper, this value can be set to
            14 ~ 24 depending on your problem size. Defaults to 15.
        base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
        finest_resolution (int, optional): finest grid resolution. According to
            the paper, this value can be set to 512 ~ 524288. Defaults to 512.
        """
        super().__init__()
        self.dim = dim
        self.n_levels = cfg.n_levels
        self.n_features_per_level = cfg.n_features_per_level
        self.log2_hashmap_size = cfg.log2_hashmap_size
        self.base_resolution = cfg.base_resolution
        self.finest_resolution = cfg.finest_resolution
        b = np.exp((np.log(self.finest_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1))
        print("b",b)
        # from paper eq (3)
        self.grid_encoder =tcnn.Encoding(
                n_input_dims=self.dim,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": self.n_levels,
                    "per_level_scale": b,
                    "n_features_per_level":self.n_features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_resolution,
                    "interpolation": "linear",
                },dtype=torch.float32
                )
        #nn.init.uniform_(self.grid_encoder.weight, a=-0.01, b=0.01)
        self.grid_in_dim = self.grid_encoder.n_output_dims
        self.simple_mlp = Simple_MLP(self.grid_in_dim, out_dim, cfg.mlp_units).to('cuda')

    def forward(self, x: torch.Tensor):
        #print(x.shape,self.dim)
        x = x/2 + 0.5
        a = self.grid_encoder(x)
        return self.simple_mlp(a)
    


