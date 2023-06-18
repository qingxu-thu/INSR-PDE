import torch
import torch.nn as nn
import numpy as np
# from pytorch3d.ops import knn_points,knn_gather
from tensorboardX import SummaryWriter
import os
import shutil

def get_network(cfg, in_features, out_features):
    if cfg.network == 'siren':
        return MLP(in_features, out_features, cfg.num_hidden_layers,
            cfg.hidden_features, nonlinearity=cfg.nonlinearity)
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
    x_o = torch.where((x>=(-5/4)&(x<-3/4)),.5+torch.sin(2*torch.pi*x)/2,x_o)
    x_o = torch.where((x>=(-3/4)&(x<3/4)),.1,x_o)
    x_o = torch.where((x>=(3/4)&(x<5/4)),.5-torch.sin(2*torch.pi*x)/2,x_o)
    x_o = x_o[...,0] * x_o[...,1]
    return x_o 

def PoU_simple(x):
    x_o = torch.zeros_like(x)
    x_o = torch.where((x>=(-1)&(x<=1)),.1,x_o)
    return x_o

# this is a heavy implementation
class Random_Basis_Function(object):
    # input for the layer is set for [-1,1]
    def  __init__(self,cfg):
        self.cfg = cfg
        num_per_point_feature = cfg.num_per_point_feature
        num_time_feature = cfg.num_time_feature
        num_spatial_basis = cfg.num_spatial_basis
        num_spatial_basis_pos = cfg.num_spatial_basis_pos
        band_width = cfg.band_width
        variable_num = cfg.variable_num
        dim = cfg.dim
        self.basis_point = torch.randn((num_spatial_basis_pos,dim))
        self.band_width = band_width
        self.spatial_A = torch.randn((num_time_feature,num_spatial_basis,variable_num,num_per_point_feature,dim))
        self.time_A = torch.randn((num_time_feature,num_spatial_basis,variable_num,num_per_point_feature))
        self.bias = torch.randn((num_time_feature,num_spatial_basis,variable_num,num_per_point_feature))
        self.PoU = PoU_simple
        self.x_process = x_process
        self.t_process = t_process
        self.non_linear = nn.Sigmoid()
        

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

    def derive_t_operator(self,x,norm):
        # for Sigmoid for highest order 2
        L_t = self.spatial_t[None,...] * (1-x)*x
        return L_t

    def cal_homo(self,x,t,boundary=None,norm=None):
        x_ = self.x_process(x,self.basis_point,self.band_width)
        # We need to implement a KNN version for the martix recon
        # Here we just use a meshed version (Maybe Hashing??)
        sptail_val = torch.einsum('tnejd,qnd->qtnej',self.spatial_A,x_)
        t_ =self.t_process(t)
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
        t_ =self.t_process(t)
        x_weight,t_weight = self.get_sparsity(x_,t_)

        time_val = torch.einsum('tnej,qt->qtnej',self.time_A,t_)
        ot = self.non_linear(sptail_val+time_val+self.bias)
        ot = torch.einsum('tnej,qtnej->qtne',u,ot,ot)
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
    

    
    
        
# # KNN spatial implementation for a higher number 
# class Random_Basis_Function(object):
#     # input for the layer is set for [-1,1]
#     def  __init__(self,num_per_point_feature,num_time_feature,num_spatial_basis,num_spatial_basis_pos,band_width,dim=2):
#         self.basis_point = torch.randn((num_spatial_basis_pos,dim))
#         self.band_width = band_width
#         self.spatial_A = torch.randn((num_time_feature,num_spatial_basis,num_per_point_feature,dim))
#         self.time_A = torch.randn((num_time_feature,num_spatial_basis,num_per_point_feature))
#         self.bias = torch.randn((num_time_feature,num_spatial_basis,num_per_point_feature))
#         self.PoU = PoU_simple
#         self.x_process = x_process
#         self.t_process = t_process
#         self.non_linear = nn.Sigmoid()


#     # we only search for the spatial discretization but time is divided equally.
#     def neighbor_search(self,x_):
#         _,idx = knn_points(x_,self.basis_point,K=4)
#         p_reduce = knn_gather(self.basis_point,idx)
#         x_ = x_process(x_,p_reduce,self.bandwidth)
#         return x_,idx

#     # implement kd tree for neighbor search 

#     # 