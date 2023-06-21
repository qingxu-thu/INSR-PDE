import os
import argparse
from vortex import Vortex_L


# create experiment config containing all hyperparameters
cfg = argparse.ArgumentParser(add_help=False)

cfg.rho = 1000
cfg.internal_v = 8
cfg.variable_list = [2,3]
cfg.time_num = 10
cfg.colloation_pts_num  = 1000
cfg.boundary_num = 400
cfg.output_path = './results' 
cfg.gravity = 9.8
cfg.num_per_point_feature = 16
cfg.time_length = 1
cfg.num_spatial_basis = 400
cfg.variable_num = 3
cfg.dim = 2
cfg.device = 'cuda'
cfg.band_width = 0.01
cfg.log_dir = './log'
cfg.n_timesteps = 100000
cfg.neighbor_K  = 6
cfg.vis_resolution = 100
model = Vortex_L(cfg)

for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    #model.train()
    #model._vis_velocity()
    model.matrix_solver()

