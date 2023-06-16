import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, sample_random, sample_uniform, sample_boundary2D_separate
from base import Random_Basis_Function
from cg_batch import cg_batch

class Vortex(Random_Basis_Function):
    def  __init__(self,cfg):
        super(Vortex,self).__init__(cfg.num_per_point_feature,cfg.num_time_feature,cfg.num_spatial_basis,cfg.num_spatial_basis_pos,cfg.band_width,cfg.dim)
        self.cfg = cfg
        self.rho = cfg.rho
        self.u_dim = 2

        
    def eqn(self,x,t,norm):
        L1,L2,Lt,B1,ot = self.cal_homo(x,t,self.boundary,norm)
        
        LHS_1 = self.rho * L1[...,:self.u_dim]*ot + self.rho * Lt[...,:self.u_dim] + L1[...,self.u_dim:]
        RHS_1 = torch.ones_like(ot) * self.cfg.gravity * self.rho
        LHS_2 = torch.einsum('qtnjdd->qtnj',L2[...,:self.u_dim,:self.u_dim])
        RHS_2 = torch.zeros_like(LHS_2)
        LHS_3 = B1
        RHS_3 = torch.zeros_like(LHS_3)
        LHS_4 = ot[self.p_boundary,...,self.u_dim:]
        RHS_4 = torch.zeros_like(LHS_4)
        LHS_5 = ot[self.u_boundary,...,:self.u_dim]
        RHS_5 = torch.ones_like(LHS_5)
        RHS_5[...,1] = 0
        RHS_5[...,0] = 8
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
        u_k,_ = cg_batch(LHS.reshape(pts,-1),RHS.reshape(pts,-1))
        return u_k
    
    def vis_vecolity(self):