import torch
import torch.nn as nn
import numpy as np


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

def x_procss(x,x_0,bandwidth):
    x = (x[:,None,:]-x_0[None,:,:])/bandwidth
    return x

# Here we can choose different types of bump function
# x.shape == x_0.shape
def PoU(x):
    x_o = torch.zeros_like(x)
    x_o = torch.where((x>=(-5/4)&(x<-3/4)),.5+torch.sin(2*torch.pi*x)/2,x_o)
    x_o = torch.where((x>=(-3/4)&(x<3/4)),.1,x_o)
    x_o = torch.where((x>=(3/4)&(x<5/4)),.5-torch.sin(2*torch.pi*x)/2,x_o)
    return x_o

def PoU_simple(x):
    x_o = torch.zeros_like(x)
    x_o = torch.where((x>=(-1)&(x<=1)),.1,x_o)
    return x_o

class Random_Basis_Function(object):
    # input for the layer is set for [-1,1]
    def  __init__(self,num_per_point_feature,num_time_feature,num_spatial_basis,num_spatial_basis_pos,band_width,dim=2):
        self.basis_point = torch.randn((num_spatial_basis_pos,dim))
        self.band_width = band_width
        self.spatial_A = torch.randn((num_spatial_basis,num_per_point_feature,dim))
        self.spatial_B = torch.randn((num_spatial_basis,num_per_point_feature))
        self.time_A = torch.randn((num_time_feature,num_spatial_basis,num_per_point_feature,dim))
        self.time_B = torch.randn((num_time_feature,num_spatial_basis,num_per_point_feature))
        self.PoU = PoU
        self.x_process = x_procss
        self.non_linear = nn.Sigmoid()
        

    def derive_order_operator(self,x,norm):
        # for Sigmoid for highest order 2
        L_1 = self.spatial_A[None,...] * (1-x)*x
        L_2 = self.spatial_A[None,...] *(1-2*x) * L_1
        B_1 = torch.einsum('ijk,ak->aijk',L_1,norm)
        return L_1, L_2, B_1

    def cal_homo(self,x,t):
        x_ = self.x_process(x,self.basis_point,self.band_width)
        x_ = torch.einsum('')
        x = 
        
        
        