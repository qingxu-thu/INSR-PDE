import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, sample_random, sample_uniform, sample_boundary2D_separate
from base import Random_Basis_Function,Random_Basis_Function_L
from .cg_batch import cg_batch
from .visualize import draw_vector_field2D, draw_scalar_field2D, draw_curl, draw_magnitude, save_numpy_img, save_figure
from base import gradient, divergence, laplace, jacobian
# from torchsparsegradutils import sparse_triangular_solve, sparse_generic_solve
# from torchsparsegradutils.utils import linear_cg, minres, rand_sparse, rand_sparse_tri





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
                           [[1 - epsilon, 1 + epsilon], [-1, 1]],
                           [[-1 - epsilon, -1 + epsilon], [-1, 1]],
                           ]
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

    def construct_and_solve(self,resolution,boundary_num):
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


class batch_input(torch.utils.data.Dataset):
    def __init__(self,pts,t,inner_pts,dir_bound,neu_bound,u_left,init_pts,norm):
        print(pts.shape,t.shape)
        self.pts = pts
        self.t = t
        bz = self.pts.shape[0]
        #print(bz,norm.shape)
        self.flag = torch.zeros((bz,5)).bool().to(self.pts.device)
        self.flag[inner_pts,0] = 1
        self.flag[dir_bound,1] = 1
        self.flag[neu_bound,2] = 1
        self.flag[u_left,3] = 1
        self.flag[init_pts,4] = 1
        
        self.norm = torch.zeros((bz,norm.shape[1])).to(self.pts.device)
        self.norm[self.flag[:,2],:] = norm
        
    def __len__(self):
        return self.pts.shape[0]

    def __getitem__(self,index):
        
        return self.pts[index],self.t[index],self.flag[index,0],self.flag[index,1],self.flag[index,2],self.flag[index,3],self.flag[index,4],self.norm[index]

class Vortex_L(Random_Basis_Function_L):
    def  __init__(self,cfg):
        super(Vortex_L,self).__init__(cfg)
        self.cfg = cfg
        self.rho = cfg.rho
        self.internal_v = cfg.internal_v
        self.variable_list = cfg.variable_list
        self.time_num = self.cfg.time_num
        self.boundary_num = self.cfg.boundary_num
        self.device = cfg.device
        self.colloation_pts_num = self.cfg.colloation_pts_num
        self._create_tb(cfg.output_path)
        self.total_samples,self.t,self.norm = self.process_input()
        self.num_process()
        self.optim = torch.optim.Adam([self.u_], lr = 0.1)

    def process_boundary(self,N,epsilon=1e-4):
        boundary_ranges = [[[-1, 1], [-1 - epsilon, -1 + epsilon]],
                           [[-1, 1], [1 - epsilon, 1 + epsilon]],
                           [[1 - epsilon, 1 + epsilon], [-1, 1]],
                           [[-1 - epsilon, -1 + epsilon], [-1, 1]],
                           ]
        coords = []
        Kp = 0
        for i,bound in enumerate(boundary_ranges):
            x_b, y_b = bound
            points = torch.empty(N // 4, 2).to(self.device)
            points[:, 0] = torch.rand(N // 4) * (x_b[1] - x_b[0]) + x_b[0]
            points[:, 1] = torch.rand(N // 4) * (y_b[1] - y_b[0]) + y_b[0]
            coords.append(points)
            Kp += points.shape[0]
            if i==0:
                norm = torch.zeros(N // 4, 2)
                norm[:,1] += 1
            elif i==1:
                self.u_boundary = Kp
                norm = torch.cat([norm,-norm],dim=0)
            elif i==3:
                self.u_boundary_left = Kp
            elif i==2:
                self.p_boundary = Kp
        coords = torch.cat(coords, dim=0)
        self.boundary_num = coords.shape[0]
        return coords, norm

    def process_time(self,time,end_time,spatial_pts,norm):
        length, dim = spatial_pts.shape
        t =  torch.linspace(0,end_time,time,device=self.device).unsqueeze(1).repeat(1,length).reshape(-1,1).requires_grad_(True)
        spatial_pts = spatial_pts.unsqueeze(0).repeat(time,1,1)
        norm = norm.unsqueeze(0).repeat(time,1,1)[1:,:,:].reshape(-1,dim).to(self.device)
        spatial_pts = spatial_pts.reshape(-1,dim)
        self.init_pts = length-(self.u_boundary_left-self.p_boundary)

        return spatial_pts,t,norm

    def process_input(self):
        ####----collocation-pts----u_boundary---u_boundary_left---p_boundary----
        grid_samples = sample_random(self.colloation_pts_num, 2, device=self.device).requires_grad_(True)
        boundary_samples,norm = self.process_boundary(self.boundary_num)
        total_samples = torch.cat([grid_samples,boundary_samples],dim=0)
        total_samples,t,norm = self.process_time(self.time_num,self.time_length,total_samples,norm)
        return total_samples,t,norm

    def mse_loss(self,x,y):
        max_x = torch.abs(x).max()
        if max_x.item()==0:
            return torch.mean((x-y)**2)*0.0
        else:
            return torch.mean((x-y)**2)/max_x

    def num_process(self):
        points = torch.linspace(0,self.time_num*(self.colloation_pts_num+self.boundary_num)-1,self.time_num*(self.colloation_pts_num+self.boundary_num),device=self.device).reshape(self.time_num,-1)
        self.inner_pts = points[1:,:self.colloation_pts_num].reshape(-1).long()
        self.dir_bound = points[1:,self.colloation_pts_num+self.u_boundary:self.colloation_pts_num+self.u_boundary+self.p_boundary].reshape(-1).long()
        self.neu_bound = points[1:,self.colloation_pts_num:self.colloation_pts_num+self.u_boundary].reshape(-1).long()
        self.u_left = points[:,self.colloation_pts_num+self.boundary_num-self.u_boundary_left:].reshape(-1).long()
        self.init_pts = points[0,:self.colloation_pts_num+self.boundary_num-self.u_boundary_left].reshape(-1).long()

    def train(self):
        
        dataset = batch_input(self.total_samples,self.t,self.inner_pts,self.dir_bound,self.neu_bound,self.u_left,self.init_pts,self.norm)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1024,shuffle=True)
        for i,data in enumerate(dataloader):
            self.optim.zero_grad()
            x,t,inner_pts,neu_bound,dir_bound,u_boundary_left,init_pts, norm = data

            #print(torch.sum(inner_pts)+0.0)
            loss = self.train_step_batch(x,t,norm,inner_pts,neu_bound,dir_bound,u_boundary_left,init_pts)
            print("loss",loss)
            loss.backward()
            self.optim.step()

    # Actually, we need to use PCG.
    def train_step(self,x,t,norm):
        loss = 0

        L1,L2,Lt,ot = self.forward(x,t)
        
        # L1:qed,Lt:qe,B1:qe,ot:qe 
        LHS_1 = self.rho * torch.einsum('qed,qd->qe', L1[self.inner_pts,:self.variable_list[0],:], \
                                        ot[self.inner_pts,:self.variable_list[0]]) \
                                         + self.rho * Lt[self.inner_pts,:self.variable_list[0]].reshape_as(ot[self.inner_pts,:self.variable_list[0]]) \
                                        + L1[self.inner_pts,self.variable_list[0]:self.variable_list[1],:].reshape_as(ot[self.inner_pts,:self.variable_list[0]])
        RHS_1 = torch.ones_like(LHS_1) * self.cfg.gravity * self.rho
        
        LHS_2 = torch.einsum('qdd->qd',L1[self.inner_pts,...,:self.variable_list[0],:])
        RHS_2 = torch.zeros_like(LHS_2)
        #(Some problem!!!)
        LHS_3 = torch.einsum('qe,qe->q',ot[self.neu_bound,:self.variable_list[0]], norm)
        RHS_3 = torch.zeros_like(LHS_3)

        LHS_4 = ot[self.dir_bound,self.variable_list[0]:self.variable_list[1]]
        RHS_4 = torch.zeros_like(LHS_4)
        LHS_5 = ot[self.u_boundary_left,:self.variable_list[0]]
        RHS_5 = torch.ones_like(LHS_5)
        RHS_5[...,1] = 0
        RHS_5[...,0] = self.internal_v
        LHS_6 = ot[self.init_pts]
        RHS_6 = torch.zeros_like(LHS_6)
        LHS = [LHS_1,LHS_2,LHS_3,LHS_4,LHS_5,LHS_6]
        RHS = [RHS_1,RHS_2,RHS_3,RHS_4,RHS_5,RHS_6]
        k = 0
        for i,j in zip(LHS,RHS):
            loss += self.mse_loss(i,j) 
        return loss
    
    def train_step_batch(self,x,t,norm,inner_pts,neu_bound,dir_bound,u_boundary_left,init_pts):
        loss = 0
        L1,L2,Lt,ot = self.forward(x,t)
        # L1:qed,Lt:qe,B1:qe,ot:qe 
        LHS_1 = self.rho * torch.einsum('qed,qd->qe', L1[inner_pts,:self.variable_list[0],:], \
                                        ot[inner_pts,:self.variable_list[0]]) \
                                         + self.rho * Lt[inner_pts,:self.variable_list[0]].reshape_as(ot[inner_pts,:self.variable_list[0]]) \
                                        + L1[inner_pts,self.variable_list[0]:self.variable_list[1],:].reshape_as(ot[inner_pts,:self.variable_list[0]])
        RHS_1 = torch.ones_like(LHS_1) * self.cfg.gravity * self.rho
        
        LHS_2 = torch.einsum('qdd->qd',L1[inner_pts,...,:self.variable_list[0],:])
        RHS_2 = torch.zeros_like(LHS_2)
        #(Some problem!!!)
        #print(ot[neu_bound,:self.variable_list[0]].shape,norm.shape)
        # if ot[neu_bound,:self.variable_list[0]].shape[0]==0:
        #     LHS_3 = torch.Tensor([0.0])
        # else:
        LHS_3 = torch.einsum('qe,qe->q',ot[neu_bound,:self.variable_list[0]], norm[neu_bound])
        RHS_3 = torch.zeros_like(LHS_3)

        LHS_4 = ot[dir_bound,self.variable_list[0]:self.variable_list[1]]
        RHS_4 = torch.zeros_like(LHS_4)
        LHS_5 = ot[u_boundary_left,:self.variable_list[0]]
        RHS_5 = torch.ones_like(LHS_5)
        RHS_5[...,1] = 0
        RHS_5[...,0] = self.internal_v
        LHS_6 = ot[init_pts]
        RHS_6 = torch.zeros_like(LHS_6)
        LHS = [LHS_1,LHS_2,LHS_3,LHS_4,LHS_5,LHS_6]
        RHS = [RHS_1,RHS_2,RHS_3,RHS_4,RHS_5,RHS_6]
        k = 0
        for i,j in zip(LHS,RHS):
            #print(LHS_1,RHS_1)
            loss += self.mse_loss(i,j) 
        return loss



    def expand_idx(self,idx,e0,e1):
        # idx:qh
        j = self.num_per_point_feature
        idx_list = []
        for i in range(e0,e1):
            K = self.idx_box.reshape(-1,self.variable_num,j)[:,i,:].expand(idx.shape[0],-1,-1)
            idx = self.idx.expand(-1,-1,j)
            idx = torch.gather(K,1,idx).reshape(idx.shape[0],-1)
            idx_list.append(idx)
        idx = torch.cat(idx_list,dim=0)
        return idx

    def expand_idx_norm(self,idx,e0,e1):
        # idx:qh
        j = self.num_per_point_feature
        K = self.idx_box.reshape(-1,self.variable_num,j)[:,e0:e1,:].expand(idx.shape[0],-1,-1,-1)
        idx = self.idx.expand(-1,-1,e1-e0,j)
        idx = torch.gather(K,1,idx).reshape(idx.shape[0],-1)
        return idx

    

    def sparse_matrix_recon(self,x,t,norm):
        #TODO: NEED TO FIX THE 1,2's q position
        #TODO: the problem for idx need to refixed for number
        L1,L2,Lt,ot,idx = self.matrix_ids(x,t)
        j = self.num_per_point_feature
        ## Sparse Matrix Shape: (tnej) * (qq) * (tnej)
        # Row shape: (tnej) (q * q)
        # L1:qhejd,Lt:qhej,B1:qhej,ot:qhej u: tnej
        # idx qh
        # LHS1: qhej
        LHS_1 = self.rho * torch.einsum('qhejd,qhej->qhej', L1[self.inner_pts,...,:self.variable_list[0],:], ot[self.inner_pts,...,:self.variable_list[0]]) + self.rho * Lt[self.inner_pts,...,:self.variable_list[0],:] + L1[self.inner_pts,...,self.variable_list[0]:self.variable_list[1],:]
        # RHS1: qej
        RHS_1 = torch.ones_like(ot[self.inner_pts,0,:,0]) * self.cfg.gravity * self.rho
        # LHS: (qe)hj
        LHS_1 = LHS_1.permute(0,2,1,3).reshape(self.inner_pts.shape[0]*self.variable_list[0],-1)
        RHS_1 = RHS_1.reshape(self.inner_pts.shape[0]*self.variable_list[0])
        idx1 = self.expand_idx(idx[self.inner_pts],0,self.variable_list[0]) #(qe)hj
        dimk = torch.linspace(0,LHS_1.shape[0]-1,1).unsqueeze(-1).repeat(LHS_1.shape[1])
        idx1 = torch.stack([idx1,dimk],dim=2).reshape(-1,2)

        # LHS2: qhej
        LHS_2 = torch.einsum('qhdjd->qhej',L1[self.inner_pts,...,:self.variable_list[0],:])
        RHS_2 = torch.zeros_like(LHS_2[self.inner_pts,0,:,0])
        LHS_2 = LHS_2.permute(0,2,1,3).reshape(self.inner_pts.shape[0]*self.variable_list[0],-1)
        RHS_2 = RHS_2.reshape(self.inner_pts.shape[0]*self.variable_list[0])
        idx2 = self.expand_idx(idx[self.inner_pts],0,self.variable_list[0]) #(qe)hj
        dimk = torch.linspace(LHS_1.shape[0],LHS_2.shape[0]-1,1).unsqueeze(-1).repeat(LHS_2.shape[1])
        idx2 = torch.stack([idx2,dimk],dim=2).reshape(-1,2)
        
        # LHS3: q'hej (Some problem!!!)
        LHS_3 = torch.einsum('qhdj,qd->qhdj',ot[self.neu_bound,:self.variable_list[0]], norm)
        RHS_3 = torch.zeros_like(LHS_3[self.neu_bound,0,:,0])
        LHS_3 = LHS_3.permute(0,2,1,3).reshape(self.neu_bound.shape[0],-1)
        RHS_3 = RHS_3.reshape(self.neu_bound.shape[0])
        idx3 = self.expand_idx_norm(idx[self.neu_bound],0,self.variable_list[0]) #(q)ehj
        dimk = torch.linspace(LHS_2.shape[0],LHS_3.shape[0]-1,1).unsqueeze(-1).repeat(LHS_3.shape[1])
        idx3 = torch.stack([idx3,dimk],dim=2).reshape(-1,2)

        # LHS4: q'hej
        LHS_4 = ot[self.dir_bound,...,self.variable_list[0]:self.variable_list[1],:]
        RHS_4 = torch.zeros_like(LHS_4[self.dir_bound,0,:,0])
        LHS_4 = LHS_4.permute(0,2,1,3).reshape(self.dir_bound.shape[0]*(self.variable_list[1]-self.variable_list[0]),-1)
        RHS_4 = RHS_4.reshape(self.dir_bound.shape[0]*(self.variable_list[1]-self.variable_list[0]))
        idx4 = self.expand_idx(idx[self.dir_bound],self.variable_list[0],self.variable_list[1]) #(qe)hj
        dimk = torch.linspace(LHS_3.shape[0],LHS_4.shape[0]-1,1).unsqueeze(-1).repeat(LHS_4.shape[1])
        idx4 = torch.stack([idx4,dimk],dim=2).reshape(-1,2)

        # LHS5: q'hej
        LHS_5 = ot[self.u_boundary_left,...,:self.variable_list[0],:]
        RHS_5 = torch.ones_like(LHS_5[self.u_boundary_left,0,:,0])
        RHS_5[...,1] = 0
        RHS_5[...,0] = self.internal_v
        LHS_5 = LHS_5.permute(0,2,1,3).reshape(self.u_boundary_left.shape[0]*self.variable_list[0],-1)
        RHS_5 = RHS_5.reshape(self.u_boundary_left.shape[0]*self.variable_list[0])
        idx5 = self.expand_idx(idx[self.u_boundary_left],0,self.variable_list[0]) #(qe)hj
        dimk = torch.linspace(LHS_4.shape[0],LHS_5.shape[0]-1,1).unsqueeze(-1).repeat(LHS_5.shape[1])
        idx5 = torch.stack([idx5,dimk],dim=2).reshape(-1,2)

        # LHS6: q'hej
        LHS_6 = ot[self.init_pts]
        RHS_6 = torch.zeros_like(LHS_6[self.init_pts,0,:,0])
        LHS_5 = LHS_5.permute(0,2,1,3).reshape(self.init_pts.shape[0]*self.variable_list[1],-1)
        RHS_5 = RHS_5.reshape(self.init_pts.shape[0]*self.variable_list[1])
        idx6 = self.expand_idx(idx[self.init_pts],0,self.variable_list[1]) #(qe)hj
        dimk = torch.linspace(LHS_5.shape[0],LHS_6.shape[0]-1,1).unsqueeze(-1).repeat(LHS_6.shape[1])
        idx6 = torch.stack([idx6,dimk],dim=2).reshape(-1,2)

        LHS = [LHS_1,LHS_2,LHS_3,LHS_4,LHS_5,LHS_6]
        RHS = [RHS_1,RHS_2,RHS_3,RHS_4,RHS_5,RHS_6]
        idx = [idx1,idx2,idx3,idx4,idx5,idx6]


        #LHS: q?h?ej
        q = sum([i.shape[0]*LHS.shape[2] for i in LHS])
        h = self.num_time_feature*self.num_spatial_basis*self.variable_num*self.num_per_point_feature
        LHS = torch.cat([i.reshape(-1) for i in LHS],dim=0)
        A = torch.sparse_coo_tensor(indices=idx,values=LHS,size=(q,h))
        b = torch.cat(RHS,dim=0)
        return A,b

    def sparse_solver(self,A,b):
        A = A.transpose()@A
        b = A.transpose()@A
        out = torch.randint(1, (b.shape[0],), dtype=torch.float64)
        torch.linalg.solve(A, b, out)
        return out

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
