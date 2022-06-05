import math

import torch

import data

world_mat_object = torch.tensor([
    [0.5085, 0.3226, 0.7984, 0.0000],
    [-0.3479, 0.9251, -0.1522, 0.0000],
    [-0.7877, -0.2003, 0.5826, 0.3384],
    [0.0000, 0.0000, 0.0000, 1.0000]
])

world_mat_inv = torch.tensor([
    [0.4019, 0.9157, 0.0000, 0.3359],
    [-0.1932, 0.0848, 0.9775, -1.0227],
    [0.8951, -0.3928, 0.2110, -7.0748],
    [-0.0000, 0.0000, -0.0000, 1.0000]
])

proj = torch.tensor([
    [2.1875, 0.0000, 0.0000, 0.0000],
    [0.0000, 3.8889, 0.0000, 0.0000],
    [0.0000, 0.0000, -1.0020, -0.2002],
    [0.0000, 0.0000, -1.0000, 0.0000]
])


def generate_roation(phi_x, phi_y, phi_z):
    def Rx(theta):
        return torch.tensor([[1, 0, 0],
                             [0, math.cos(theta), -math.sin(theta)],
                             [0, math.sin(theta), math.cos(theta)]])

    def Ry(theta):
        return torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                             [0, 1, 0],
                             [-math.sin(theta), 0, math.cos(theta)]])

    def Rz(theta):
        return torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                             [math.sin(theta), math.cos(theta), 0],
                             [0, 0, 1]])

    return Rz(phi_z) @ Ry(phi_y) @ Rx(phi_x)


def rotate_pc(pc, rx, ry, rz):
    rotation = generate_roation(rx, ry, rz)
    rotated = pc.clone()
    rotated[:, :3] = rotated[:, :3] @ rotation.T
    if rotated.shape[-1] == 6:
        rotated[:, 3:] = rotated[:, 3:] @ rotation.T
    return rotated


def draw_pc(pc: torch.Tensor, res=(540, 960), radius=5, timer=None, dy=0, scale=1):
    xyz = pc[:, :3]
    xyz -= xyz.mean(dim=0)
    t_scale = xyz.norm(dim=-1).max()
    xyz /= t_scale
    xyz *= scale

    xyz[:, -1] += xyz[:, -1].min()

    n, _ = xyz.shape

    if timer is not None:
        with timer('project'):
            xyz_pad = torch.cat([xyz, torch.ones_like(pc[:, :1])], dim=-1)
            xyz_local = xyz_pad @ world_mat_inv.T
            distances = -xyz_local[:, 2]

            projected = xyz_local @ proj.T
            projected = projected / projected[:, 3:4]
            projected = projected[:, :3]

            u_pix = ((projected[0] + 1) / 2) * res[1]
            v_pix = ((projected[1] + 1) / 2) * res[0] + dy

        with timer('z-buffer'):
            z_buffer = data.scatter(u_pix, v_pix, distances, res, radius=radius)[:, :]
    else:
        xyz_pad = torch.cat([xyz, torch.ones_like(pc[:, :1])], dim=-1)
        xyz_local = xyz_pad @ world_mat_inv.T
        distances = -xyz_local[:, 2]

        projected = xyz_local @ proj.T
        projected = projected / projected[:, 3:4]
        projected = projected[:, :3]

        u_pix = ((projected[:, 0] + 1) / 2) * res[1]
        v_pix = ((projected[:, 1] + 1) / 2) * res[0] + dy

        z_buffer = data.scatter(u_pix.numpy(), v_pix.numpy(), distances, res, radius=radius)[:, :]

    z_buffer = z_buffer[data.RANGES[0][0]: data.RANGES[0][1], :]
    z_buffer = z_buffer[:, data.RANGES[1][0]:data.RANGES[1][1]]
    z_buffer = data.resize(z_buffer)
    return z_buffer
