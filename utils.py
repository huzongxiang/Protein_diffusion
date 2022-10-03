# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:25:46 2022

@author: huzongxiang
"""


import math
import tensorflow as tf
from typing import Union, Sequence
from tensor import Tensor


def assert_err_values(tensor: Tensor) -> Tensor:
    """
    Check error values like Nan, inf et.

    Parameters
    ----------
    tensor : Tensor
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """

    return tf.compat.v1.check_numerics(tensor, "non number")


def extract(t: Tensor, tensor: Tensor) -> Tensor:
    """
    Reshape t according to tensor, make t can multipy with tensor

    Parameters
    ----------
    t : Tensor
        DESCRIPTION.
    tensor : Tensor
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """

    shape = (tf.shape(t)[0], ) + (1, ) * (tf.shape(tensor).shape[0] - 1)

    return tf.reshape(t, shape)


def gravity_to_zero(tensor: Tensor, graph_indices: Tensor) -> Tensor:
    """
    p(x) is invariant about x_zero_gravity
    x_zero_gravity is x substract its center of gravity

    Parameters
    ----------
    tensor : Tensor
        DESCRIPTION.
    graph_indices : Tensor
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    
    gravity = tf.math.segment_mean(tensor, graph_indices)
    tensor_zero_gravity = tensor - tf.gather(gravity, graph_indices)
    
    return tensor_zero_gravity


def assert_gravity_to_zero(tensor: Tensor, graph_indices: Tensor, eps: float=1e-8):
    """
    Check whether or not tensor's gravity is zero.

    Parameters
    ----------
    tensor : Tensor
        DESCRIPTION.
    graph_indices : Tensor
        DESCRIPTION.
    eps : float, optional
        DESCRIPTION. The default is 1e-10.

    Returns
    -------
    None.

    """
    
    largest_value = tf.math.segment_max(tensor, graph_indices)
    batch_error = tf.math.segment_sum(tensor, graph_indices)
    error = batch_error / (largest_value + eps)
    mean_error = tf.math.abs(tf.math.reduce_mean(error))
    assert mean_error < 1e-2, f'mean gravity is not zero, relative_error {mean_error}'


def gaussian_kl(mu_1: Tensor, sigma_1: Tensor,
                mu_2: Union[Tensor, None]=None, sigma_2: Union[Tensor, None]=None) -> Tensor:
    """
    Calculate KL divergence of two gaussian distributions.

    Parameters
    ----------
    mu_1 : Tensor
        DESCRIPTION.
    sigma_1 : Tensor
        DESCRIPTION.
    mu_2 : Union[Tensor, None], optional
        DESCRIPTION. The default is None.
    sigma_2 : Union[Tensor, None], optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    
    if (mu_2 is None and sigma_2 is not None) or (mu_2 is not None and sigma_2 is None):
        raise ValueError("error mu2 and sigma_2")
    if mu_2 is None and sigma_2 is None:
        mu_2 = tf.zeros_like(mu_1)
        sigma_2 = tf.ones_like(sigma_1)
    kl = 0.5 * (2 * tf.math.log(sigma_2 / sigma_1) + (sigma_1 + tf.math.squared_difference(mu_1, mu_2)) / sigma_2 - 1)
    return tf.math.reduce_sum(kl, axis=-1)
    

def gaussian_kl_subspace(mu_1: Tensor, sigma_1: Tensor,
                         mu_2: Union[Tensor, None]=None, sigma_2: Union[Tensor, None]=None,
                         d_sub: Union[Tensor, None]=None) -> Tensor:
    """
    Calculate KL divergence of two gaussian distributions in subspace.

    Parameters
    ----------
    mu_1 : Tensor
        DESCRIPTION.
    sigma_1 : Tensor
        DESCRIPTION.
    mu_2 : Union[Tensor, None], optional
        DESCRIPTION. The default is None.
    sigma_2 : Union[Tensor, None], optional
        DESCRIPTION. The default is None.
    d_sub : Union[Tensor, None], optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    
    if (mu_2 is None and sigma_2 is not None) or (mu_2 is not None and sigma_2 is None):
        raise ValueError("error mu2 and sigma_2")
    if mu_2 is None and sigma_2 is None:
        mu_2 = tf.zeros_like(mu_1)
        sigma_2 = tf.ones_like(sigma_1)
    kl_sub = d_sub * 0.5 * (2 * tf.math.log(sigma_2 / sigma_1) + (sigma_1 + tf.math.squared_difference(mu_1, mu_2)) / sigma_2 - 1)
    return tf.reduce_sum(kl_sub, axis=-1)


def standard_cdf(tensor: Tensor) -> Tensor:
    """
    Cumulative Distribution Function of a standard normal distribution.

    Parameters
    ----------
    tensor : Tensor
        DESCRIPTION.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    
    return 0.5 * (1 + tf.math.erf(tensor / math.sqrt(2.)))


def cosine_beta_schedule(timesteps:int, s: float= 0.008) -> Tensor:
    """
    Cosine schedule

    Parameters
    ----------
    timesteps : int
        DESCRIPTION.
    s : float, optional
        DESCRIPTION. The default is 0.008.

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    
    steps = timesteps + 1
    x = tf.linspace(0, timesteps, steps)
    alphas_cumprod = tf.math.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0, 0.9999)