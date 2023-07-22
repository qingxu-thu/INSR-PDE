import torch


def sample_boundary2D_separate_list(N, side, epsilon=1e-4, device='cpu'):
    """sample boundary points within a small range. NOTE: random samples, not uniform"""
    bound_list = [[[-1, -1 + epsilon], [-1, 1]], # left
                    [[1 - epsilon, 1], [-1, 1]], # right
                    [[-1, 1], [-1, -1 + epsilon]], # down
                    [[-1, 1], [1 - epsilon, 1]]] # upper

    boundary_ranges = [bound_list[i] for i in side]
    coords = []
    for bound in boundary_ranges:
        x_b, y_b = bound
        points = torch.empty(N // 2, 2, device=device)
        points[:, 0] = torch.rand(N // 2, device=device) * (x_b[1] - x_b[0]) + x_b[0]
        points[:, 1] = torch.rand(N // 2, device=device) * (y_b[1] - y_b[0]) + y_b[0]
        coords.append(points)
    coords = torch.cat(coords, dim=0)
    return coords


def circle(x, r=0.282, center=torch.Tensor([0., 0.])):
    dx = x - center
    return torch.sqrt(torch.sum((dx)**2, axis=1)) - r

def sample_sdf(N,r,shift,device):
    angle = torch.rand(N, device=device) * 2 * torch.pi
    coord = torch.stack([torch.cos(angle)*r,torch.sin(angle)*r],axis=-1) + shift 
    norm = torch.stack([torch.cos(angle),torch.sin(angle)],axis=-1)
    return coord, norm

def sample_mlp():
    pass

def pts_proj(samples,r,shift):
    mask = torch.where(torch.sqrt(torch.sum((samples-shift)**2, axis=-1)) - r > 0,0,1).detach().bool()
    #rand = torch.rand_like(samples[mask])*1e-5
    samples[mask] = (samples[mask]-shift)/(1e-20+torch.norm(samples[mask]-shift,dim=-1))[:,None] * r + shift
    return samples


def sampling_sdf_field(N,sdim,device,r,shift,eps=1e-5):
    samples = torch.rand(N, sdim, device=device) * 2 - 1
    r = r + eps
    return samples
