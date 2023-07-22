import torch
import torch.nn.functional as F

def c2v(v):
    return torch.stack((v[:,0].real, v[:,0].imag, v[:,1].real, v[:,1].imag),dim=-1)

def v2c(v):
    v = torch.view_as_complex(v.reshape(-1,2)).reshape(-1,2)
    return v

def v2a(v):
    a, b, c, d = v[:,0], v[:,1], v[:,2], v[:,3]
    alpha = torch.sqrt(a * a + b * b)
    beta = torch.atan2(b, a)
    gamma = torch.atan2(d, c)
    return torch.stack((alpha, beta, gamma),dim=-1)

def q2v(v):
    angle = torch.norm(v,dim=-1)
    axis = v / angle[:,None]
    q = angle * axis
    return q

def q2v(q):
    return q

def v2q(v):
    return v

def normalize(psi):
    norm = torch.sqrt(torch.norm(psi[:,0]) + torch.norm(psi[:,1]))
    psi[0] /= norm
    psi[1] /= norm
    return psi

def vel_to_psi_c(vel, pos):
    bz = vel.shape[0]
    psi = torch.view_as_complex(torch.stack([torch.tensor([1., 0.]), torch.tensor([.1, 0.])])).repeat(bz,1)
    psi = normalize(psi)
    phase = (vel * pos).sum(-1)
    for i in range(2):
        psi[:,i] *= torch.exp(1j * phase)
    return psi

def vel_to_psi_c_with_init(vel, pos, init_psi):
    psi = v2c(init_psi)
    print(psi)
    phase = (vel * pos).sum(-1)
    for i in range(2):
        psi[:,i] *= torch.exp(1j * phase)
    return psi

def vel_to_psi(vel, pos):
    return c2v(vel_to_psi_c(vel, pos))

# Example usage:
vel = torch.tensor([[0.5, 0.2],[0.5, 0.2]])
pos = torch.tensor([[1.0, 2.0],[0.5, 0.2]])
init_psi = torch.tensor([[1.0, 0.1, 2.0, 3.0],[1.0, 0.1, 2.0, 3.0]])


if __name__=='main':
    result1 = vel_to_psi_c_with_init(vel, pos, init_psi)
    print(result1)
    result = vel_to_psi(vel, pos)
    print("Result:", result)