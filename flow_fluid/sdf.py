import torch
import torch.func.vmap as vmap

# sdf primitives
# Reference: https://iquilezles.org/articles/distfunctions2d/

def _blobby_cross(pos, he=0.45):
    pos = torch.abs(pos)
    pos = torch.Tensor([torch.abs(pos[0] - pos[1]), 1. - pos[0] - pos[1]
                    ]) / torch.sqrt(2)

    p = (he - pos[1] - 0.25 / he) / (6. * he)
    q = pos[0] / (he * he * 16)
    h = q * q - p * p * p

    r = torch.where(h > 0., torch.sqrt(h), torch.sqrt(p))
    x = torch.where(
        h > 0.,
        torch.pow(q + r, 1. / 3.) -
        torch.pow(torch.abs(q - r), 1. / 3.) * torch.sign(r - q),
        2. * r * torch.cos(torch.arccos(q / (p * r)) / 3.))

    x = torch.minimum(x, torch.sqrt(2.) / 2.)
    z = torch.array([x, he * (1. - 2. * x * x)]) - pos
    return torch.linalg.norm(z) * torch.sign(z[1])


def blobby_cross(x, he=0.45):
    return vmap(_blobby_cross, (0, None))(x, he)


def _hexgram(p, r=0.22):
    xy = torch.Tensor([-0.5, 0.8660254038])
    yx = torch.Tensor([0.8660254038, -0.5])
    z = 0.5773502692
    w = 1.7320508076

    p = torch.abs(p)
    p = p - 2. * torch.minimum(torch.dot(xy, p), 0.) * xy
    p = p - 2. * torch.minimum(torch.dot(yx, p), 0.) * yx
    p = p - torch.array([torch.clip(p[0], r * z, r * w), r])
    return torch.linalg.norm(p) * torch.sign(p[1])


def hexgram(x, r=0.22):
    return vmap(_hexgram, (0, None))(x, r)


def _equilateral_triangle(p, r=0.4):
    k = torch.sqrt(3.)

    p = p.at[0].set(torch.abs(p[0]) - r)
    p = p.at[1].set(p[1] + r / k)
    p = torch.where((p[0] + k * p[1]) > 0.,
                  torch.array([p[0] - k * p[1], -k * p[0] - p[1]]) / 2., p)
    p = p.at[0].add(-torch.clip(p[0], -2. * r, 0.))
    return -torch.linalg.norm(p) * torch.sign(p[1])


def equilateral_triangle(x, r=0.4):
    return vmap(_equilateral_triangle, (0, None))(x, r)


def rot2d(theta):
    return torch.Tensor([[torch.cos(theta), -torch.sin(theta)],
                      [torch.sin(theta), torch.cos(theta)]])


# https://github.com/ml-for-gp/jaxgptoolbox/blob/main/general/sdf_circle.py

def circle(x, r=0.282, center=torch.array([0., 0.])):
    dx = x - center
    return torch.sqrt(torch.sum((dx)**2, axis=1)) - r


# training sdfs

def star(x):
    return 0.5 * hexgram(2.0 * x, 0.6)


def two_stars(x):
    return torch.minimum(
        0.5 * hexgram(2.0 * x - 0.4, 0.6), 0.25 * hexgram(
            (2. + 4.0 * x) @ rot2d(torch.pi / 4).T, 0.6))


def two_starts_inv(x):
    return two_stars(-x)


def triangle(x):
    return 0.5 * equilateral_triangle(2.0 * x, 1.0)


def _annular(x, p, sdf, d, c=0):
    return torch.abs(sdf(x, p) + c) - d


def triangle_ring(x):
    return 0.5 * vmap(_annular, (0, None, None, None))(
        2.0 * x, 1.0, _equilateral_triangle, 0.25)


def triangle_ring_rot90(x):
    return 0.5 * vmap(_annular, (0, None, None, None))(
        2.0 * x @ rot2d(-1 * torch.pi / 2).T, 1.0, _equilateral_triangle, 0.25)


def clover(x):
    return 0.5 * (blobby_cross(2.0 * x, 1.2) - 0.5)


def clover_ring(x):
    return 0.5 * vmap(_annular, (0, None, None, None, None))(
        2.0 * x, 1.2, _blobby_cross, 0.15, -0.5)


def clover_ring_fat_rot90(x):
    return 0.5 * vmap(_annular, (0, None, None, None, None))(
        2.0 * x @ rot2d(torch.pi / 4).T, 1.2, _blobby_cross, 0.15, -0.5)
