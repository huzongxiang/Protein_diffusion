# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 17:20:38 2022

@author: huzongxiang
"""


import math
import numpy as np
from typing import List, Union, Sequence
from tensor import Tensor
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from egnn_diffusion import Egnn_diffusion
from utils import (extract,
                   gravity_to_zero,
                   # assert_gravity_to_zero,
                   gaussian_kl,
                   gaussian_kl_subspace,
                   standard_cdf,
                   assert_err_values,
                   )


def linear_schedule(timesteps):
    """
    Linear schedule

    Parameters
    ----------
    timesteps : TYPE
        DESCRIPTION.

    Returns
    -------
    alphas_cumprod : TYPE
        DESCRIPTION.

    """
    
    scale = 1000 / (timesteps + 1)
    beta_start = scale * 0.0001
    beta_end = scale * 0.01
    betas = tf.linspace(beta_start, beta_end, timesteps + 1)
    alphas2 = 1. - betas
    alphas2 = tf.math.cumprod(alphas2, axis=0)
    
    return alphas2


def cosine_schedule(timesteps:int, s: float= 0.008) -> Tensor:
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
    
    steps = timesteps + 2
    x = tf.linspace(0, timesteps, steps)
    alphas_cumprod = tf.math.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = tf.clip_by_value(betas, 0, 0.9999)
    
    alphas2 = 1. - betas
    alphas2 = np.cumprod(alphas2, axis=0)
    
    return alphas2


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = tf.concat([tf.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = tf.clip_by_value(alphas_step, clip_value, 1.)
    alphas2 = tf.math.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=2.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = tf.linspace(0, steps, steps)
    alphas2 = tf.cast((1 - tf.math.pow(x / steps, power))**2, dtype=tf.float32)

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


class NoiseSchedule(layers.Layer):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, timesteps, noise_schedule="cosine", power=2):
        super().__init__()
        
        assert noise_schedule in ["linear", "cosine", "polynomial"]
        
        self.timesteps = timesteps

        if noise_schedule == "polynomial":
            alphas2 = polynomial_schedule(timesteps, power=power)
        if noise_schedule == 'cosine':
            alphas2 = cosine_schedule(timesteps)
        else:
            alphas2 = linear_schedule(timesteps)
        
        sigmas2 = 1 - alphas2

        log_alphas2 = tf.math.log(alphas2)
        log_sigmas2 = tf.math.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = tf.cast(-log_alphas2_to_sigmas2, dtype=tf.float32)


    def call(self, t):
        t_int = tf.cast(t * self.timesteps, dtype=tf.int32)
        t_int = tf.where(t_int < 0, 0, t_int)
        return tf.gather(self.gamma, t_int)


class PositiveLinear(layers.Layer):
    
    def __init__(self, out_features: int, **kwargs) -> None:
        super().__init__( **kwargs)

        self.dense = layers.Dense(out_features, kernel_constraint=NonNeg())


    def call(self, inputs: Tensor) -> Tensor:
        return self.dense(inputs)


class Gamma(layers.Layer):
    """
    Gamma function is a monotonic neural network with input, e.g., gamma(t) > gamma(s) for t > s.
    This implementation as described in the paper
    Variational Diffusion Models (Kingma. et al. 2021):
    (http://arxiv.org/abs/2107.00630)
    """
    def __init__(self) -> None:
        super().__init__()

        self.l1 = PositiveLinear(1)
        self.l2 = PositiveLinear(1024)
        self.l3 = PositiveLinear(1)

        self.gamma_min = -10
        self.gamma_max = 20


    def gamma_tilde(self, t: Tensor):
        l1_t = self.l1(t)
        return l1_t + self.l3(tf.math.sigmoid(self.l2(l1_t)))


    def call(self, t: Tensor) -> Sequence:

        # assert tf.shape(t)[-1] == 1, f"inputs a row vector with shape ({tf.shape(t)[-1]},), should be column vector with shape ({tf.shape(t)[-1]}, 1)"

        zeros, ones = tf.zeros_like(t), tf.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * normalized_gamma

        return gamma