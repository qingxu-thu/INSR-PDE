import torch
import torch.nn as nn
import numpy as np
from pytorch3d.ops import knn_points,knn_gather
from tensorboardX import SummaryWriter
import os
import shutil
import math
from .diff_ops import *


from .hash_encoding import MultiResHashGrid

def get_network(cfg, in_features, out_features):
    if cfg.network == 'siren':
        return MLP(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features, nonlinearity=cfg.nonlinearity)
    elif cfg.network == 'hashgrid':
        return MultiResHashGrid(in_features,out_features,cfg)
    else:
        raise NotImplementedError
    


############################### SIREN ################################
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


def x_process(x,x_0,bandwidth):
    x = (x[:,None,:]-x_0[None,:,:])/bandwidth
    return x

def t_process(t,t_0,bandwidth):
    t = (t[:,None]-t_0[None,:])/bandwidth
    return t

# Here we can choose different types of bump function
# x.shape == x_0.shape
def PoU(x):
    x_o = torch.zeros_like(x)
    #print(x)
    a = 1/4
    x_o = torch.where(torch.logical_and(x>=(-(1+a)),(x<-(1-a))),.5+torch.sin(torch.pi/(4*a)*x)/2,x_o)
    x_o = torch.where(torch.logical_and(x>=(-(1-a)),(x<(1-a))),1,x_o)
    x_o = torch.where(torch.logical_and(x>=((1-a)),(x<(1+a))),.5-torch.sin(2*torch.pi/(4*a)*x)/2,x_o)
    # x_o = x_o[...,0] * x_o[...,1]
    return x_o 

def PoU_simple(x):
    x_o = torch.zeros_like(x)
    #print(x[432])
    x_o = torch.where(torch.logical_and(x>=(-1),(x<=1)),1,x_o)
    return x_o

# this is a heavy implementation
class Random_Basis_Function(object):
    # input for the layer is set for [-1,1]
    def  __init__(self,cfg):
        self.cfg = cfg
        self.num_per_point_feature = cfg.num_per_point_feature
        self.num_time_feature = cfg.time_num
        self.time_length = self.cfg.time_length
        self.num_spatial_basis = cfg.num_spatial_basis
        self.band_width = cfg.band_width
        self.variable_num = cfg.variable_num
        self.dim = cfg.dim
        self.device = cfg.device
        self.basis_point,self.basis_time = self.generate_basis(self.num_spatial_basis,self.num_time_feature,self.time_length,self.dim)
        self.band_width = cfg.band_width
        self.spatial_A = torch.randn((self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature,self.dim))
        self.time_A = torch.randn((self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature))
        self.bias = torch.randn((self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature))
        self.PoU = PoU_simple
        self.x_process = x_process
        self.t_process = t_process
        self.non_linear = nn.Sigmoid()
        self.tb = None


    def generate_basis(self,pos_num,time_num,end_time,dim):
        resolution = int(math.pow(pos_num,1/dim))
        coords = torch.linspace(0.5, resolution - 0.5, resolution, device=self.device) / resolution * 2 - 1
        coords = torch.stack(torch.meshgrid([coords] * dim, indexing='ij'), dim=-1)
        coords = coords.reshape(resolution**dim, dim)
        length, dim = coords.shape[-1]
        t =  torch.linspace(0,end_time,time_num).unsqueeze(1).repeat(1,length).reshape(-1,1)
        coords = coords.unsqeeze(0).repeat(time_num,1,1)
        coords = coords.reshape(-1,dim)
        return coords,t

    def derive_order_operator(self,x_,boundary=None,norm=None):
        # for Sigmoid for highest order 2
        # L_1 = self.spatial_A[None,...] * (1-x)*x
        # L_2 = self.spatial_A[None,...] *(1-2*x) * L_1
        # B_1 = torch.einsum('ijk,ak->aijk',L_1,norm)
        L_1 = torch.einsum('tnejd,qtnej->qtnejd',self.spatial_A,(1-x_)*x_)
        L_2 = torch.einsum('tnejd,qtnej,tnejd->qtnejdd',self.spatial_A,(1-x_)*x_,self.spatial_A)
        B_1 = None
        L_t = torch.einsum('tnej,qtnej->qtnej',self.time_A,(1-x_)*x_)
        if norm is not None:
            B_1 = torch.einsum('qtnejd,qd->qtnejd',L_1[boundary],norm)
        return L_1, L_2, L_t, B_1


    def cal_homo(self,x,t,boundary=None,norm=None):
        x_ = self.x_process(x,self.basis_point,self.band_width)
        # We need to implement a KNN version for the martix recon
        # Here we just use a meshed version (Maybe Hashing??)
        sptail_val = torch.einsum('tnejd,qnd->qtnej',self.spatial_A,x_)
        t_ =self.t_process(t,self.basis_time,self.time_length/self.num_time_feature)
        x_weight,t_weight = self.get_sparsity(x_,t_)

        time_val = torch.einsum('tnej,qt->qtnej',self.time_A,t_)
        ot = self.non_linear(sptail_val+time_val+self.bias)
        L1,L2,Lt,B1 = self.derive_order_operator(ot,boundary,norm)
        ot = torch.einsum('qn,qt,qtnej->qtnej',x_weight,t_weight,ot)
        L1 = torch.einsum('qn,qt,qtnejd->qtnejd',x_weight,t_weight,L1)
        L2 = torch.einsum('qn,qt,qtnejdd->qtnejdd',x_weight,t_weight,L2)
        Lt = torch.einsum('qn,qt,qtnej->qtnej',x_weight,t_weight,Lt)
        if norm is not None:
            B1 = torch.einsum('qn,qt,qtnejd->qtnej',x_weight[boundary],t_weight[boundary],B1)
        return L1,L2,Lt,B1,ot
    
    def inference(self,x,t,u):
        # ot: qtnej
        # u: tnej
        x_ = self.x_process(x,self.basis_point,self.band_width)
        # We need to implement a KNN version for the martix recon
        # Here we just use a meshed version (Maybe Hashing??)
        sptail_val = torch.einsum('tnejd,qnd->qtnej',self.spatial_A,x_)
        t_ =self.t_process(t,self.basis_time,self.time_length/self.num_time_feature)
        x_weight,t_weight = self.get_sparsity(x_,t_)

        time_val = torch.einsum('tnej,qt->qtnej',self.time_A,t_)
        ot = self.non_linear(sptail_val+time_val+self.bias)
        ot = torch.einsum('tnej,qtnej->qtne',u,ot)
        ot = torch.einsum('qn,qt,qtne->qe',x_weight,t_weight,ot)
        return ot

    def get_sparsity(self,x,t):
        x_= self.PoU(x)
        t_ = self.PoU(t)
        return x_,t_

    def _create_tb(self, name, overwrite=True):
        """create tensorboard log"""
        self.log_path = os.path.join(self.cfg.log_dir, name)
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)
        if self.tb is not None:
            self.tb.close()
        self.tb = SummaryWriter(self.log_path)
    


def gather_use(A,idx):
    A = A.expand(idx.shape[0],-1,-1)
    idx = idx.expand(-1,-1,A.shape[1])
    A = torch.gather(A,1,idx)
    return A

# KNN spatial implementation for a higher number 
class Random_Basis_Function_L(object):
    # input for the layer is set for [-1,1]
    def  __init__(self,cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.num_per_point_feature = cfg.num_per_point_feature
        self.num_time_feature = cfg.time_num
        self.time_length = self.cfg.time_length
        self.time_band_width = self.time_length/self.num_per_point_feature
        self.num_spatial_basis = cfg.num_spatial_basis
        self.variable_num = cfg.variable_num
        self.dim = cfg.dim
        self.device = cfg.device
        self.neighbor_K = cfg.neighbor_K
        self.device = self.cfg.device
        self.basis_point,self.basis_time = self.generate_basis(self.num_spatial_basis,self.num_time_feature,self.time_length,self.dim)
        #self.band_width = cfg.band_width
        self.spatial_A = torch.rand((self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature,self.dim),device=self.device)
        self.spatial_A = self.band_width * 2*(self.spatial_A-0.5)

        self.low_basis_A = torch.rand((self.num_time_feature,self.variable_num,self.num_per_point_feature,self.dim),device=self.device)
        self.low_basis_A = 2*(self.low_basis_A-0.5)
        self.bias = torch.rand((self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature),device=self.device)
        self.bias = 2*(self.bias-0.5)
        self.low_bias_A = torch.rand((self.num_time_feature,self.variable_num,self.num_per_point_feature),device=self.device)
        self.low_bias_A = 2*(self.low_bias_A-0.5)
        a = torch.rand(self.num_time_feature,self.num_spatial_basis,self.variable_num,self.num_per_point_feature,device=self.device)
        a = 2*(a-0.5)
        self.u_ = torch.nn.Parameter(a)
        a = torch.rand(self.num_time_feature,self.variable_num,self.num_per_point_feature,device=self.device)
        a = self.band_width * 2*(a-0.5)   
        self.global_u_ =  torch.nn.Parameter(a)    
        self.idx_box = torch.linspace(0,self.num_spatial_basis*self.variable_num*self.num_per_point_feature-1,self.num_spatial_basis*self.variable_num*self.num_per_point_feature,device=self.device)
        self.idx_box = self.idx_box.reshape(self.num_spatial_basis,self.variable_num,self.num_per_point_feature)
        self.global_idx_box = self.num_spatial_basis*self.variable_num*self.num_per_point_feature + torch.linspace(0,self.variable_num*self.num_per_point_feature-1,self.variable_num*self.num_per_point_feature,device=self.device)
        self.global_idx_box = self.global_idx_box.reshape(self.variable_num,self.num_per_point_feature)
        self.PoU = PoU
        self.non_linear = nn.Tanh()
        self.tb = None

    def x_process(self,x,x_0,bandwidth):
        #print(x.shape,x_0.shape)
        x = (x[:,:,None,:]-x_0[:,:,:,:])/bandwidth
        return x

    def t_process(self,t,t_0,bandwidth):
        t = (t[:,None]-t_0[:,:])/bandwidth
        return t

    def generate_basis(self,pos_num,time_num,end_time,dim):
        resolution = int(math.pow(pos_num,1/dim))
        self.band_width = 1/resolution
        
        coords = torch.linspace(0.5, resolution - 0.5, resolution, device=self.device) / resolution * 2 - 1
        coords = torch.stack(torch.meshgrid([coords] * dim, indexing='ij'), dim=-1)
        coords = coords.reshape(resolution**dim, dim)
        
        length, dim = coords.shape
        self.num_spatial_basis = coords.shape[0]
        #print(self.num_spatial_basis)
        t =  torch.linspace(0,end_time,time_num, device=self.device).unsqueeze(1).repeat(1,length)
        coords = coords.unsqueeze(0).repeat(time_num,1,1)
        #coords = coords.reshape(-1,dim)
        #print(coords.shape,t.shape)
        return coords,t

    def _create_tb(self, name, overwrite=True):
        """create tensorboard log"""
        self.log_path = os.path.join(self.cfg.log_dir, name)
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)
        if self.tb is not None:
            self.tb.close()
        self.tb = SummaryWriter(self.log_path)

    # we only search for the spatial discretization but time is divided equally.
    def neighbor_search(self,x_,t_):
        # Need to expand self.basis_pts
        bz = x_.shape[0]
        dim = x_.shape[-1]
        tdim = t_.shape[-1]
        xt_ = torch.cat([x_,t_],dim=1).unsqueeze(0)
        
        plex = torch.cat([self.basis_point,self.basis_time*(self.band_width/self.time_band_width*1)],dim=1).unsqueeze(0)
        xt_[:,-1] *= (self.band_width/self.time_band_width)*1
        #x_ = x_.unsqueeze(1)
        _,idx,_ = knn_points(xt_,plex,K=self.neighbor_K,return_nn=False)
        p_reduce = knn_gather(plex,idx)
        
        # Might be some problem with x_process
        #—p_reduce: bz,1,k,tdim
        x_0 = p_reduce[...,:-1]
        t_0 = p_reduce[...,-1]*(self.time_band_width/self.band_width)/1
        x_0 = x_0.reshape(bz,self.neighbor_K,dim)
        t_0 = t_0.reshape(bz,self.neighbor_K,tdim)
        #print(x_,x_0,self.band_width)
        x_ = self.x_process(x_,x_0,self.band_width)
        t_ = self.t_process(t_,t_0,self.time_band_width).squeeze(-1)
        idx = idx.squeeze(0)
        return x_,t_,idx

    def neighbor_search_spatial(self,x_):
        #print(x_.shape,self.basis_point.shape)
        bz = x_.shape[0]
        pts_num = x_.shape[1]
        dim = x_.shape[-1]
        #tdim = t_.shape[-1]
        #xt_ = torch.cat([x_],dim=1).unsqueeze(0)
        xt_ = x_
        plex = self.basis_point
        #print(xt_.shape,plex.shape)
        #xt_[:,-1] *= (self.band_width/self.time_band_width)*1
        #x_ = x_.unsqueeze(1)
        #print(x_.shape,plex.shape)
        #print(xt_.shape,plex.shape)
        _,idx,_ = knn_points(xt_,plex,K=self.neighbor_K,return_nn=False)
        p_reduce = knn_gather(plex,idx)
        #print(p_reduce.shape)
        #print(p_reduce.shape)
        # Might be some problem with x_process
        # —p_reduce: bz,1,k,tdim
        # x_0 = p_reduce[...,:-1]
        # t_0 = p_reduce[...,-1]*(self.time_band_width/self.band_width)/1
        x_0 = p_reduce.reshape(bz,pts_num,self.neighbor_K,dim)
        # t_0 = t_0.reshape(bz,self.neighbor_K,tdim)
        # print(x_.shape,x_0.shape)
        # print(x_,x_0,self.band_width)
        #print(x_0.shape,x_.shape)
        x_ = self.x_process(x_,x_0,self.band_width)
        return x_,idx
    

    def neighbor_search_single(self,x_,time):
        bz = x_.shape[0]
        pts_num = x_.shape[1]
        dim = x_.shape[-1]
        xt_ = x_
        plex = self.basis_point[time].unsqueeze(0)
        #print(x_.shape,plex.shape)
        _,idx,_ = knn_points(xt_,plex,K=self.neighbor_K,return_nn=False)
        p_reduce = knn_gather(plex,idx)
        x_0 = p_reduce.reshape(bz,pts_num,self.neighbor_K,dim)
        x_ = self.x_process(x_,x_0,self.band_width)
        idx = idx.squeeze(0)
        x_ = x_.squeeze(0)
        return x_,idx

    def forward(self,x,t):
        # Actually, we do not need to do all this on the gpus
        x_,t_,idx = self.neighbor_search(x,t)
        h = idx.shape[1]
        bz = idx.shape[0]
        total_ = self.num_time_feature*self.num_spatial_basis
        # Size is a problem here, we might abondon time
        A_process = self.spatial_A.reshape(total_,-1)
        A_process = A_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,A_process.shape[-1]) 
        A_process = torch.gather(A_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature,self.dim)
        t_process_ = self.time_A.reshape(total_,-1)
        t_process_ = t_process_.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,t_process_.shape[-1])
        t_process_ = torch.gather(t_process_,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        bias_process = self.bias.reshape(total_,-1)
        bias_process = bias_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,bias_process.shape[-1])
        bias_process = torch.gather(bias_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        u_process = self.u_.reshape(total_,-1)
        u_process = u_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,u_process.shape[-1])
        u_process = torch.gather(u_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        #print(t_process_.shape,x_.shape,t_.shape)

        sptail_val = torch.einsum('qhejd,qhd->qhej',A_process,x_)
        time_val = torch.einsum('qhej,qh->qhej',t_process_,t_)
        ot = self.non_linear(sptail_val+time_val+bias_process)
        #A,t,b: qhej; u:qhej
        x_weight,t_weight = self.get_sparsity(x_,t_)
        #x_weight, t_weight: qh,qh
        #print(x_weight.shape,t_weight.shape)
        ot = x_weight[...,0][...,None,None]*x_weight[...,1][...,None,None]*t_weight[...,None,None]*u_process*ot
        ot = torch.sum(torch.sum(ot,dim=-1),dim=1)
        L1,_ = jacobian(ot, x)
        # TODO: We need A HESSIAN!!!!!!!!!!!!!!
        L2 = None
        Lt,_ = jacobian(ot, t)
        # B1 = None
        # if norm is not None:
        #     B1 = L1[boundary] * norm.unsqeeze(1)
        return L1,L2,Lt,ot
    
    def semi_lagrangian_advection(self,samples,prev_u,time):
        samples = samples[time].clone().detach().requires_grad_(True)
        
        backtracked_position = samples - prev_u * self.cfg.dt
        #print(backtracked_position.shape,prev_u.shape)
        backtracked_position = torch.clamp(backtracked_position, min=-1.0, max=1.0)
        # we need a neighbor mechanism and derive the speed
        x_,idx = self.neighbor_search_single(backtracked_position.unsqueeze(0),time)
        total_ = self.num_time_feature*self.num_spatial_basis
        h = idx.shape[1]
        bz = idx.shape[0]
        
        A_process = self.spatial_A[time-1].reshape(self.num_spatial_basis,-1)
        A_process = A_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,A_process.shape[-1]) 
        #print(A_process.shape,idx_.shape)
        A_process = torch.gather(A_process,1,idx_)
        A_process = A_process.reshape(-1,h,self.variable_num,self.num_per_point_feature,self.dim)
        bias_process = self.bias[time-1].reshape(self.num_spatial_basis,-1)
        bias_process = bias_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,bias_process.shape[-1]) 
        bias_process = torch.gather(bias_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        #print(x_.shape)
        u_process = self.u_[time-1].reshape(self.num_spatial_basis,-1)
        u_process = u_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,u_process.shape[-1])
        u_process = torch.gather(u_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        sptail_val = torch.einsum('qhejd,qhd->qhej',A_process,x_)
        #time_val = torch.einsum('qhej,qh->qhej',t_process_,t_)
        ot = self.non_linear(sptail_val+bias_process)

        x_weight = self.PoU(x_)
        x_weight = x_weight[...,0] * x_weight[...,1]
        ot = ot * x_weight[...,None,None]

        u_current = torch.einsum('qhej,qhej->qe',ot,u_process)
        L1,_ = jacobian(u_current[:,:2], samples)
        L1 = torch.einsum('qdd->q',L1)
        return prev_u[:,:2], L1, samples 
        
    
    def matrix_ids(self,x,t):
        x_,t_,idx = self.neighbor_search(x,t)
        h = idx.shape[1]
        bz = idx.shape[0]
        total_ = self.num_time_feature*self.num_spatial_basis
        # Size is a problem here, we might abondon time
        A_process = self.spatial_A.reshape(total_,-1)
        A_process = A_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,A_process.shape[-1]) 
        A_process = torch.gather(A_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature,self.dim)
        t_process_ = self.time_A.reshape(total_,-1)
        t_process_ = t_process_.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,t_process_.shape[-1])
        t_process_ = torch.gather(t_process_,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        bias_process = self.bias.reshape(total_,-1)
        bias_process = bias_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,bias_process.shape[-1])
        bias_process = torch.gather(bias_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        # u_process = self.u_.reshape(total_,-1)
        # u_process = torch.gather(u_process,idx).reshape(-1,self.variable_num,self.num_per_point_feature)
        sptail_val = torch.einsum('qhejd,qhd->qhej',A_process,x_)
        time_val = torch.einsum('qhej,qh->qhej',t_process_,t_)
        ot = self.non_linear(sptail_val+time_val+bias_process)
        #A,t,b: qhej; u:qhej
        x_weight,t_weight = self.get_sparsity(x_,t_)
        

        #print("x_weight",x_weight)
        
        #x_weight, t_weight: qh,qh'
        # something should be wrong here
        ot =  x_weight[...,0][...,None,None]*x_weight[...,1][...,None,None]*t_weight[...,None,None]*ot
        # print(x_weight)
        
        q_,h_,e_,j_ = ot.shape
        ot = ot.reshape(ot.shape[0],-1)

        L1,_ = jacobian(ot, x)

        L1 = L1.reshape(q_,h_,e_,j_,-1)
        # L1_p = L1.reshape(q_,-1)
        # L1_p = torch.sum(torch.abs(L1_p),dim=-1)
        # print(torch.argwhere(L1_p==0))
        #L2 = hessian(ot.unsqueeze(-1), x_.unsqueeze(1).repeat(1,ot.shape[1],1))
        # Actually, we do not have Hessian
        L2 = None
        Lt,_ = jacobian(ot, t)
        Lt = Lt.reshape(q_,h_,e_,j_,-1)
        ot = ot.reshape(q_,h_,e_,j_ )
        # B1 = None
        # if norm is not None:
        #     B1 = L1[boundary] * norm.unsqeeze(1)
        #print("ot",L1[0])
        return L1,L2,Lt,ot,idx   

    def get_sparsity(self,x,t):
        #print(x,t)
        x_= self.PoU(x)
        t_ = self.PoU(t)
        #print(x_.reshape(-1),t_.reshape(-1))
        return x_,t_
    
    def inference(self,x,t):
        # x_: Q * dim
        x_,t_,idx = self.neighbor_search(x,t)
        h = idx.shape[1]
        bz = idx.shape[0]
        total_ = self.num_time_feature*self.num_spatial_basis
        # Size is a problem here, we might abondon time
        A_process = self.spatial_A.reshape(total_,-1)
        A_process = A_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,A_process.shape[-1]) 
        A_process = torch.gather(A_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature,self.dim)
        t_process_ = self.time_A.reshape(total_,-1)
        t_process_ = t_process_.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,t_process_.shape[-1])
        t_process_ = torch.gather(t_process_,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        bias_process = self.bias.reshape(total_,-1)
        bias_process = bias_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,bias_process.shape[-1])
        bias_process = torch.gather(bias_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        u_process = self.u_.reshape(total_,-1)
        u_process = u_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,u_process.shape[-1])
        u_process = torch.gather(u_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)

        sptail_val = torch.einsum('qhejd,qhd->qhej',A_process,x_)
        time_val = torch.einsum('qhej,qh->qhej',t_process_,t_)
        ot = self.non_linear(sptail_val+time_val+bias_process)
        #A,t,b: qhej; u:qhej
        x_weight,t_weight = self.get_sparsity(x_,t_)
        #x_weight, t_weight: qh,qh
        #print(u_process.shape,)
        ot = x_weight[...,0][...,None,None]*x_weight[...,1][...,None,None]*t_weight[...,None,None]*u_process*ot
        
        # print(ot,u_process)
        ot = torch.sum(torch.sum(ot,dim=-1),dim=1)
        # ot:qe
        return ot   


    def inference_time(self,x,t):
        # x_: Q * dim
        
        x_,idx = self.neighbor_search_single(x.unsqueeze(0),t)
        #print("idx",idx.shape)
        total_ = self.num_time_feature*self.num_spatial_basis
        h = idx.shape[1]
        bz = idx.shape[0]
        #print(x.shape)
        A_process = self.spatial_A[t].reshape(self.num_spatial_basis,-1)
        A_process = A_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,A_process.shape[-1]) 
        #print(A_process.shape,idx_.shape)
        A_process = torch.gather(A_process,1,idx_)
        A_process = A_process.reshape(-1,h,self.variable_num,self.num_per_point_feature,self.dim)
        bias_process = self.bias[t].reshape(self.num_spatial_basis,-1)
        bias_process = bias_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,bias_process.shape[-1]) 
        bias_process = torch.gather(bias_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        #print(x_.shape)
        u_process = self.u_[t].reshape(self.num_spatial_basis,-1)
        #print("u_procsss",u_process)
        u_process = u_process.unsqueeze(0).expand(bz,-1,-1)
        idx_ = idx.unsqueeze(-1).expand(-1,-1,u_process.shape[-1])
        u_process = torch.gather(u_process,1,idx_).reshape(-1,h,self.variable_num,self.num_per_point_feature)
        sptail_val = torch.einsum('qhejd,qhd->qhej',A_process,x_)
        #print(sptail_val[0])
        #time_val = torch.einsum('qhej,qh->qhej',t_process_,t_)
        ot = self.non_linear(sptail_val+bias_process)
        
        x_weight = self.PoU(x_)
        x_weight = x_weight[...,0] * x_weight[...,1]
        ot = ot * x_weight[...,None,None]
        
        global_feature = self.non_linear(torch.einsum('ejd,qd->qej',self.low_basis_A[t],x) + self.low_bias_A[t,None,:,:])
        global_feature = torch.einsum('qej,ej->qe',global_feature,self.global_u_[t])


        #print(ot,u_process)
        u_current = torch.einsum('qhej,qhej->qe',ot,u_process)
        u_current = u_current
        #print(ot.shape)
        return u_current[:,:2], u_current[:,2]