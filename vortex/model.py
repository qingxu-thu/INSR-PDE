import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, sample_random, sample_uniform, sample_boundary2D_separate
from base import Random_Basis_Function
from cg_batch import cg_batch
from .visualize import draw_vector_field2D, draw_scalar_field2D, draw_curl, draw_magnitude, save_numpy_img, save_figure
from base import gradient, divergence, laplace, jacobian

class Vortex(Random_Basis_Function):
    def  __init__(self,cfg):
        super(Vortex,self).__init__(cfg)
        self.cfg = cfg
        self.rho = cfg.rho
        self.internal_v = cfg.internal_v
        self.variable_list = cfg.variable_list
        self.time_num = self.cfg.time_num
        self._create_tb(cfg.output_path)

    def process_boundary(self,N,epsilon=1e-4):
        boundary_ranges = [[[-1, 1], [-1 - epsilon, -1 + epsilon]],
                           [[-1, 1], [1 - epsilon, 1 + epsilon]],
                           [[1 - epsilon, 1 + epsilon], [-1, 1]]
                           [[-1 - epsilon, -1 + epsilon], [-1, 1]],
                           ,]
        coords = []
        for i,bound in enumerate(boundary_ranges):
            x_b, y_b = bound
            points = torch.empty(N // 4, 2)
            points[:, 0] = torch.rand(N // 4) * (x_b[1] - x_b[0]) + x_b[0]
            points[:, 1] = torch.rand(N // 4) * (y_b[1] - y_b[0]) + y_b[0]
            coords.append(points)
            if i==0:
                norm = torch.zeros(N // 4, 2)
                norm[:,1] += 1
            elif i==1:
                self.u_boundary = len(coords)
                norm = torch.cat([norm,-norm],dim=0)
            elif i==3:
                self.u_boundary_left = len(coords)
            elif i==2:
                self.p_boundary = len(coords)
        coords = torch.cat(coords, dim=0)
        return coords, norm

    def process_time(self,time,end_time,spatial_pts):
        length, dim = spatial_pts.shape[-1]
        t =  torch.linspace(0,end_time,time).unsqueeze(1).repeat(1,length).reshape(-1,1)
        spatial_pts = spatial_pts.unsqeeze(0).repeat(time,1,1)
        spatial_pts = spatial_pts.reshape(-1,dim)
        self.init_pts = length-(self.u_boundary_left-self.p_boundary)

        return spatial_pts,t

    def eqn(self,x,t,norm):
        # L1: qtnejd
        # ot: qtnej
        # Lt: qtnej
        
        L1,L2,Lt,B1,ot = self.cal_homo(x,t,self.boundary,norm)
        LHS_1 = self.rho * torch.einsum('qtnejd,qtnej->qtnjd', L1[...,:self.variable_list[0],:,:] , ot) + self.rho * Lt[...,:self.variable_list[0],:] + L1[...,self.variable_list[0]:self.variable_list[1],:]
        RHS_1 = torch.ones_like(ot) * self.cfg.gravity * self.rho

        LHS_2 = torch.einsum('qtndjd->qtnj',L1[...,:self.variable_list[0],:,:])
        RHS_2 = torch.zeros_like(LHS_2)
        LHS_3 = B1
        RHS_3 = torch.zeros_like(LHS_3)
        LHS_4 = ot[self.p_boundary,...,self.variable_list[0]:self.variable_list[1],:]
        RHS_4 = torch.zeros_like(LHS_4)
        LHS_5 = ot[self.u_boundary,...,:self.variable_list[0],:]
        RHS_5 = torch.ones_like(LHS_5)
        RHS_5[...,1] = 0
        RHS_5[...,0] = self.internal_v
        LHS_6 = ot[self.init_pts] 
        RHS_6 = torch.zeros_like(LHS_6)
        
        LHS = [LHS_1,LHS_2,LHS_3,LHS_4,LHS_5,LHS_6]
        RHS = [RHS_1,RHS_2,RHS_3,RHS_4,RHS_5,RHS_6]
        return LHS, RHS
    
    def least_square_solver(self,LHS,RHS):
        #Large matrix solver
        pts = LHS.shape[0]
        LHS = torch.einsum('qtnj,qtnj->tnjtnj',LHS,LHS)
        RHS = torch.einsum('qtnj,q->tnj',LHS,RHS)
        self.u_k,_ = cg_batch(LHS.reshape(pts,-1),RHS.reshape(pts,-1))

    def conrtruct_and_solve(self):
        grid_samples = sample_uniform(resolution, 2, device=self.device, flatten=True).requires_grad_(True)
        boundary_samples,norm = self.process_boundary(boundary_num)
        total_samples = torch.cat([grid_samples,boundary_samples],dim=0)
        total_samples,t = self.process_time(self.time_num,self.time_length,total_samples)
        LHS, RHS = self.eqn(total_samples,t,norm)
        LHS = torch.stack(LHS,dim=0)
        RHS = torch.stack(RHS,dim=0)
        self.least_square_solver(LHS,RHS)

    def sample_field(self, resolution, boundary_num, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 2, device=self.device, flatten=True).requires_grad_(True)
        boundary_samples,norm = self.process_boundary(boundary_num)
        total_samples = torch.cat([grid_samples,boundary_samples],dim=0)
        total_samples,t = self.process_time(self.time_num,self.time_length,total_samples)
        out = self.inference(total_samples,t,self.u_k)
        if return_samples:
            return out, grid_samples
        return out

    def _vis_velocity(self):
        """visualization on tb during training"""
        velos, samples = self.sample_field(self.vis_resolution, return_samples=True)
        velos = velos.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_vector_field2D(velos[:,:self.variable_list[0]], samples)
        self.tb.add_figure("velocity", fig, global_step=self.train_step)


    # def write_output(self, output_folder):
    #     grid_u, grid_samples = self.sample_field(self.vis_resolution, return_samples=True)
    #     grid_u = grid_u[:,:self.variable_list[0]]
    #     u_mag = torch.sqrt(torch.sum(grid_u ** 2, dim=-1))
    #     jaco, _ = jacobian(grid_u, grid_samples)
    #     u_curl = jaco[..., 1, 0] - jaco[..., 0, 1]
        
    #     grid_samples = grid_samples.detach().cpu().numpy()
    #     grid_u = grid_u.detach().cpu().numpy()
    #     u_mag = u_mag.detach().cpu().numpy()
    #     u_curl = u_curl.detach().cpu().numpy()

    #     fig = draw_vector_field2D(grid_u, grid_samples)
    #     save_path = os.path.join(output_folder, f"t{self.timestep:03d}_vel.png")
    #     save_figure(fig, save_path)

    #     mag_img = draw_magnitude(u_mag)
    #     save_path = os.path.join(output_folder, f"t{self.timestep:03d}_mag.png")
    #     save_numpy_img(mag_img, save_path)

    #     curl_img = draw_curl(u_curl)
    #     save_path = os.path.join(output_folder, f"t{self.timestep:03d}_curl.png")
    #     save_numpy_img(curl_img, save_path)

    #     save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npy")
    #     np.save(save_path, grid_u)