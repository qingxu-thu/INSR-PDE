import os
import argparse
from model import Vortex


# create experiment config containing all hyperparameters
cfg = argparse.ArgumentParser(add_help=False)

cfg.rho = 1000
cfg.internal_v = 8
cfg.variable_list = [0,1]
cfg.time_num = 100
cfg.output_path = './results' 
cfg.gravity = 9.8
cfg.num_per_point_feature = 100
cfg.time_length = 10
cfg.num_spatial_basis = 1000
cfg.variable_num = 3
cfg.dim = 2
cfg.device = 'cuda'
cfg.band_width = 0.1
cfg.log_dir = './log'
model = Vortex(cfg)

