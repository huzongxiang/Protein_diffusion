# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:08:13 2022

@author: huzongxiang
"""


import numpy as np


def augmented_data(residues_list, positions_list, augmented=1):
    new_residues_list = []
    new_positions_list = []
    for i, positions in enumerate(positions_list):
        for _ in range(augmented):
            rotated_positions = random_rotation(positions)
            new_positions_list.append(rotated_positions)
            new_residues_list.append(residues_list[i])
    return new_residues_list, new_positions_list
        

def random_rotation(x):
    n_dims = x.shape[-1]
    angle_range = np.pi * 2
    if n_dims == 2:
        theta = np.round(np.random.normal(0., 1., size=(1, 1)), 2) * angle_range - np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_row0 = np.concatenate([cos_theta, -sin_theta], dim=2)
        R_row1 = np.concatenate([sin_theta, cos_theta], dim=2)
        R = np.concatenate([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = np.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = np.eye(3)
        alpha = np.round(np.random.normal(0., 1., size=(1, 1)), 2) * angle_range - np.pi
        cos = np.cos(alpha)
        sin = np.sin(alpha)
        Rx[1:2, 1:2] = cos
        Rx[1:2, 2:3] = sin
        Rx[2:3, 1:2] = - sin
        Rx[2:3, 2:3] = cos

        # Build Ry
        Ry = np.eye(3)
        theta = np.round(np.random.normal(0., 1., size=(1, 1)), 2) * angle_range - np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        Ry[0:1, 0:1] = cos
        Ry[0:1, 2:3] = -sin
        Ry[2:3, 0:1] = sin
        Ry[2:3, 2:3] = cos

        # Build Rz
        Rz = np.eye(3)
        gamma = np.round(np.random.normal(0., 1., size=(1, 1)), 2) * angle_range - np.pi
        cos = np.cos(gamma)
        sin = np.sin(gamma)
        Rz[0:1, 0:1] = cos
        Rz[0:1, 1:2] = sin
        Rz[1:2, 0:1] = -sin
        Rz[1:2, 1:2] = cos

        x = np.matmul(Rx, x.transpose(1, 0))
        x = np.matmul(Ry, x)
        x = np.matmul(Rz, x).transpose(1, 0)
    else:
        raise NotImplementedError("Not implemented Error")

    return x