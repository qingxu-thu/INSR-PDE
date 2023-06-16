import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, sample_random, sample_uniform, sample_boundary2D_separate
from base import Random_Basis_Function


class Vortex(Random_Basis_Function):
    def  __init__(self,num_per_point_feature,num_time_feature,num_spatial_basis,num_spatial_basis_pos,band_width,dim=2):
        super(Vortex,self).__init__(num_per_point_feature,num_time_feature,num_spatial_basis,num_spatial_basis_pos,band_width,dim)
        