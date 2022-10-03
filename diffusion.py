# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:59:12 2022

@author: huzongxiang
"""


import math
import numpy as np
from typing import List, Union, Sequence
from tensor import Tensor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import layers
from egnn_diffusion import Egnn_diffusion
from schedule import Gamma, NoiseSchedule
from utils import (extract,
                   gravity_to_zero,
                   # assert_gravity_to_zero,
                   gaussian_kl,
                   gaussian_kl_subspace,
                   standard_cdf,
                   assert_err_values,
                   )


class VariationalGaussianDiffusion(layers.Layer):
    
    def __init__(
        self,
        dynamics:Egnn_diffusion,
        schedule:str="builtin",
        batch_size:int=32,
        node_dim:int=20,
        x_dim:int=3,
        timesteps:int=1000,
        pattern:str="noise",
        l2_loss:bool=True,
        min_signal_rate:float=0.02,
        max_signal_rate:float=0.95,
        clip_noise:bool=True,
        re_project:bool=True,
        scaling:float=0.25,
        epsilon:float=1e-4,
        num_classes:int=20,
        **kwargs,
        ):
        super().__init__(**kwargs)
        
        # assert pattern in ["denoising", "noise", "score"]

        self.batch_size = batch_size
        self.dynamics = dynamics

        self.schedule = schedule
        self.builtin_schedule = True
        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        
        if self.schedule == "builtin":
            self.builtin_schedule = True
            self.min_signal_rate = min_signal_rate
            self.max_signal_rate = max_signal_rate
        elif schedule == "learned":
            self.builtin_schedule = False
            self.gamma = Gamma()
        elif schedule == "predefined":
            self.builtin_schedule = False
            self.gamma = NoiseSchedule(timesteps=timesteps)
        else:
            raise ValueError(f"unknown schedule {schedule}, should be 'builtin', 'learned' and 'predefined'.")
        
        self.node_dim = node_dim
        self.x_dim = x_dim

        self.timesteps = timesteps
        self.pattern = pattern
        self.l2_loss = l2_loss
        self.clip_noise = clip_noise
        self.re_project = re_project
        self.scaling = scaling
        self.epsilon = epsilon
        self.num_class = num_classes

        
    def cal_num_nodes_batch(self, graph_indices: Tensor) -> Tensor:
        """

        Parameters
        ----------
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        num_nodes = tf.cast(tf.math.bincount(graph_indices), dtype=self.dtype)[:, None]

        return num_nodes


    def sample_s_t(self, graph_indices: Tensor) -> Sequence:
        """
        Sample t from a uniform distribution.

        Parameters
        ----------
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        start = 1
        if self.trainable:
            start = 0
        
        i_t = tf.random.uniform((self.batch_size, 1), start, self.timesteps + 1, dtype=tf.int32)
        i_t = tf.gather(i_t, graph_indices)
        i_s = i_t - 1

        s = tf.cast(i_s / self.timesteps, dtype=self.dtype)
        t = tf.cast(i_t / self.timesteps, dtype=self.dtype)
        
        return s, t


    def snr(self, gamma: Tensor) -> Tensor:
        """
        SNR(t) = exp(-γ(t))

        Parameters
        ----------
        gamma : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        return tf.math.exp(-gamma)


    def alpha(self, gamma: Tensor, tensor: Tensor, tensor_: Union[Tensor, None]=None) -> Tensor:
        """
        alpha = sqrt(sigmoid(-γ)) = sqrt[SNR(t)/(1 + SNR(t))]

        Parameters
        ----------
        gamma : Tensor
            DESCRIPTION.
        tensor : Tensor
            DESCRIPTION.
        tensor_ : Union[Tensor, None], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        alpha = tf.math.sqrt(tf.math.sigmoid(-gamma))
        if tensor_ is not None:
            return extract(alpha, tensor), extract(alpha, tensor_)
        
        return extract(alpha, tensor)


    def sigma(self, gamma: Tensor, tensor: Tensor, tensor_: Union[Tensor, None]=None) -> Union[Tensor, Sequence]:
        """
        sigma = sqrt(sigmoid(γ)) = sqrt[1/(1 + SNR(t))]

        Parameters
        ----------
        gamma : Tensor
            DESCRIPTION.
        tensor : Tensor
            DESCRIPTION.
        tensor_ : Union[Tensor, None], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Union[Tensor, Sequence]
            DESCRIPTION.

        """
        
        sigma = tf.math.sqrt(tf.math.sigmoid(gamma))
        if tensor_ is not None:
            return extract(sigma, tensor), extract(sigma, tensor_)
        
        return extract(sigma, tensor)


    def alpha_and_sigma(self, t: Tensor, tensor: Tensor) -> Tensor:
        """

        Parameters
        ----------
        gamma : Tensor
            DESCRIPTION.
        tensor : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        gamma = self.gamma(t)
        
        return self.alpha(gamma, tensor), self.sigma(gamma, tensor)
        
    
    def diffusion_schedule(self, t: Tensor, h: Tensor) -> Sequence:
        """
        builtin diffusion schedule
        diffusion times -> angles
        Parameters
        ----------
        t : Tensor
            DESCRIPTION.
        h : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + t * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return extract(signal_rates, h), extract(noise_rates, h)


    def sample_eps_hx_t(self, h: Tensor, x: Tensor, graph_indices: Tensor) -> Sequence:
        """

        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        # assert self.node_dim == tf.shape(h)[-1]
        # assert self.x_dim == tf.shape(x)[-1]

        eps_h = tf.random.normal(shape=tf.shape(h), mean=0.0, stddev=1.0)
        
        eps_x_gravity = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)   
        eps_x = gravity_to_zero(eps_x_gravity, graph_indices)
        
        # assert_gravity_to_zero(eps_x, graph_indices)
        
        return eps_h, eps_x
    

    def kl_prior(self, h: Tensor, x: Tensor, graph_indices: Tensor) -> Tensor:
        """
        Computes the KL divergence between q(z1|hx) and the prior p(z1).
        p(z1) = Normal(0, I), μ = 0, σ = I
        q(z1|x_h) = Normal(z1|alpha_1*x_h, sigma_1_sq*I), μ = alpha_1*x_h, σ = sigma_1_sq*I
        q(z1|x_h) = q(z1_x|x) * q(z1_h|h)
        q(z1_x|x) should be invariant with respect to x
        This term approximates zero.
        
        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        # assert self.node_dim == tf.shape(h)[-1]
        # assert self.x_dim == tf.shape(x)[-1]

        ones = tf.ones(shape=(tf.shape(graph_indices)[0], 1))

        if self.builtin_schedule:
            alpha_1, sigma_1 = self.diffusion_schedule(ones, h)
        else:
            alpha_1, sigma_1 = self.alpha_and_sigma(ones, h)

        mu_1_h = alpha_1 * h
        mu_1_x = alpha_1 * x
        
        zeros, ones = tf.zeros_like(mu_1_h), tf.ones_like(sigma_1)
        kl_h = gaussian_kl(mu_1_h, sigma_1, zeros, ones)
        kl_h = tf.math.segment_sum(kl_h, graph_indices)

        num_nodes = self.cal_num_nodes_batch(graph_indices)
        d_sub = (num_nodes - 1) * self.x_dim
        d_sub = tf.gather(d_sub, graph_indices)
        zeros, ones = tf.zeros_like(mu_1_x), tf.ones_like(sigma_1)

        kl_x = gaussian_kl_subspace(mu_1_x, sigma_1, zeros, ones, d_sub)
        kl_x = tf.math.segment_sum(kl_x, graph_indices)

        return kl_h + kl_x


    def neural(self, h: Tensor, x: Tensor, t: Tensor,
               node_indices: Tensor, pair_indices: Tensor) -> Sequence:
        """
        Equivariant graph neural network recovers the denoising h, x or predicts noise epsilon_h, epsilon_x.
        
        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        x : Tensor
            DESCRIPTION.
        t : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        h_theta, x_theta = self.dynamics([h, x, t, node_indices, pair_indices])
        
        return h_theta, x_theta


    def diffusion_loss_t(self, pred_h: Tensor, pred_x: Tensor, noisy_h: Tensor, noisy_x: Tensor,
                         t: Tensor, s: Tensor, graph_indices: Tensor) -> Tensor:
        """
        Calculate t-th loss during diffusion.
        denoising mode: L_t = [(SNR(s) − SNR(t)) ||hx − hx_θ(zt; t)||^2] / 2
        noise mode: L_t = [(exp(γ(t) − γ(s)) − 1)||epsilon − epsilon_θ(zt; t)||^2] / 2
        NOTICE: As above formula described, this term is always larger than zero !
        
        Parameters
        ----------
        h_theta : Tensor
            DESCRIPTION.
        x_theta : Tensor
            DESCRIPTION.
        h_ze : Tensor
            DESCRIPTION.
        x_ze : Tensor
            DESCRIPTION.
        gamma_t : Tensor
            DESCRIPTION.
        gamma_s : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        # assert tf.shape(h_theta)[-1] == tf.shape(h_ze)[-1] and tf.shape(x_theta)[-1] == tf.shape(x_ze)[-1]

        noisy_hx = tf.concat([noisy_h, noisy_x], axis=-1)
        pred_hx= tf.concat([pred_h, pred_x], axis=-1)
        normal = tf.math.square(noisy_hx - pred_hx)

        if self.trainable and self.l2_loss:
            weight_snr_s_t = tf.ones_like(normal)
            diffusion_loss = 0.5 * weight_snr_s_t * normal
            diffusion_loss = tf.math.segment_mean(diffusion_loss, graph_indices)

            return tf.reduce_mean(diffusion_loss, axis=-1)

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        if self.pattern == "denoising":
            weight_snr_s_t = self.snr(gamma_s) - self.snr(gamma_t)
            weight_snr_s_t = extract(weight_snr_s_t, normal)
        elif self.pattern == "noise":
            weight_snr_s_t = tf.math.expm1(gamma_t - gamma_s)
            weight_snr_s_t = extract(weight_snr_s_t, normal)
        else:
            raise ValueError(self.pattern)

        diffusion_loss = 0.5 * weight_snr_s_t * normal
        diffusion_loss = tf.math.segment_sum(diffusion_loss, graph_indices)
        
        return tf.reduce_sum(diffusion_loss, axis=-1)
    

    def log_z_x(self, x: Tensor, graph_indices: Tensor) -> Tensor:
        """
        Compute logZ 
        constant Z = [(sqrt(2*pi)*σ0/α0)]**((M-1)*n), M: num of nodes, n: dimension
        logZ = (M-1)*n*[log(σ0/α0) + log(2*pi)/2]
        σ_t^2/α_t^2 = 1/SNR(t) = exp(gamma_t)
        log(σ_t/α_t) = 0.5 * gamma_t

        Parameters
        ----------
        x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        # assert self.x_dim == tf.shape(x)[-1]

        num_nodes = self.cal_num_nodes_batch(graph_indices)
        degrees_of_freedom = (num_nodes - 1) * self.x_dim
        
        zeros = tf.zeros((tf.shape(num_nodes)[0], 1))
        gamma_0 = self.gamma(zeros)
        
        # log_0_x = log(σ_0/α_0) = 0.5 * gamma_0, it's far less than 0
        log_0_x = 0.5 * gamma_0

        return degrees_of_freedom * (log_0_x + 0.5 * tf.math.log(2 * math.pi))
    

    def log_px_given_z0(self, pred_x: Tensor, noise_x: Tensor, graph_indices: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x_theta : Tensor
            DESCRIPTION.
        x_ze : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        normal = tf.math.square(noise_x - pred_x)
        reconstruction_loss = 0.5 * normal
        reconstruction_loss = tf.math.segment_sum(reconstruction_loss, graph_indices)
        
        if self.trainable and self.l2_loss:
            return - tf.math.reduce_sum(reconstruction_loss, axis=-1)
        log_z = self.log_z_x(pred_x, graph_indices)

        return - tf.math.reduce_sum(reconstruction_loss + log_z, axis=-1)


    def log_ph_given_z0(self, h: Tensor, z0_h: Tensor, sigma_0_h: Tensor, graph_indices: Tensor) -> Tensor:
        """
        h: one_hot encoding in number of classes of h as [0, 0, 0, 1, 0 ...]
        
        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        z0_h : Tensor
            DESCRIPTION.
        sigma_0_h : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        h = h / self.scaling
        z0_h = z0_h / self.scaling
        sigma_0_h = sigma_0_h / self.scaling

        ph_given_z0_unnormal = standard_cdf((1.5 - z0_h) / sigma_0_h) - standard_cdf((0.5 - z0_h) / sigma_0_h)
        log_ph_given_z0_unnormal = tf.math.log(ph_given_z0_unnormal + self.epsilon)
        
        assert_err_values(log_ph_given_z0_unnormal)
        log_ph_given_z0_probabilities = tf.math.log_softmax(log_ph_given_z0_unnormal + self.epsilon, axis=-1)
        
        assert_err_values(log_ph_given_z0_probabilities)
        log_ph_given_z0 = tf.math.segment_sum(log_ph_given_z0_probabilities * h, graph_indices)
        
        return tf.math.reduce_sum(log_ph_given_z0, axis=-1)


    def noisy_loss(self, h: Tensor, x: Tensor,
                   pred_noisy_h: Tensor, pred_noisy_x: Tensor, graph_indices: Tensor) -> Tensor:
        """

        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        x : Tensor
            DESCRIPTION.
        pred_noisy_h : Tensor
            DESCRIPTION.
        pred_noisy_x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        noisy_hx = tf.concat([pred_noisy_h, pred_noisy_x], axis=-1)
        hx = tf.concat([h, x], axis=-1)
        loss_hx = tf.math.square(noisy_hx - hx)

        noisy_loss = tf.math.segment_mean(loss_hx, graph_indices)

        return tf.math.reduce_mean(noisy_loss, axis=-1)
        
        
    def calculate_loss(self, h: Tensor, x: Tensor,  node_indices: Tensor,
                       pair_indices: Tensor, graph_indices: Tensor) -> Tensor:
        """
        -VLB = Dkl(q(z1|hx)||p(z1)) + Dkl(q(s|t,hx)||(p(s||t))) - log(p(hx||z0))

        Parameters
        ----------
        h : Tensor
            DESCRIPTION.
        x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """

        # 0 < s, t <= 1, diffusion loss
        s, t = self.sample_s_t(graph_indices)
        
        if self.builtin_schedule:
            alpha_t, sigma_t = self.diffusion_schedule(t, h)
        else:
            alpha_t, sigma_t = self.alpha_and_sigma(t, h)
        
        noise_h, noise_x = self.sample_eps_hx_t(h, x, graph_indices)
        
        noisy_h = alpha_t * h + sigma_t * noise_h
        noisy_x = alpha_t * x + sigma_t * noise_x
        
        pred_noise_h, pred_noise_x = self.neural(noisy_h, noisy_x, t, node_indices, pair_indices)
        
        diffusion_loss_t = self.diffusion_loss_t(pred_noise_h, pred_noise_x, noise_h, noise_x,
                                                 t, s, graph_indices)
        
        if self.clip_noise:
            pred_noise_h = tf.clip_by_value(pred_noise_h,
                                            clip_value_min=(noisy_h - alpha_t) / sigma_t,
                                            clip_value_max=(noisy_h + alpha_t) / sigma_t,
                                            )

            # pred_noise_x = tf.clip_by_value(pred_noise_x,
            #                                 clip_value_min=(noisy_x - alpha_t) / sigma_t,
            #                                 clip_value_max=(noisy_x + alpha_t) / sigma_t,
            #                                 )

        pred_true_h = (noisy_h - sigma_t * pred_noise_h) / alpha_t
        pred_true_x = (noisy_x - sigma_t * pred_noise_x) / alpha_t

        pred_true_h = pred_true_h / self.scaling

        # one-hot
        pred_true_h = tf.one_hot(tf.argmax(pred_true_h, axis=-1), self.num_class)
        noisy_loss = self.noisy_loss(h, x, pred_true_h, pred_true_x, graph_indices)

        # t = 1, kl prior loss
        if self.builtin_schedule:
            kl_prior_loss = 0.0
        else:
            kl_prior_loss = self.kl_prior(h, x, graph_indices)

        # t = 0, reconstruction loss
        if self.trainable:
            reconstruction_loss_x = - self.log_px_given_z0(pred_noise_x, noise_x, graph_indices)
            reconstruction_loss_h = - self.log_ph_given_z0(h, noisy_h, sigma_t, graph_indices) 
            reconstruction_loss = reconstruction_loss_h + reconstruction_loss_x
            
            is_t0 = tf.cast((t == 0), dtype=self.dtype)
            reconstruction_loss = reconstruction_loss * is_t0
            diffusion_loss_t = (1.0 - is_t0) * diffusion_loss_t
            
            if self.trainable and self.l2_loss:
                diffusion_loss = diffusion_loss_t + reconstruction_loss
            else:
                diffusion_loss = (self.timesteps + 1) * (diffusion_loss_t + reconstruction_loss)
                
            loss = kl_prior_loss + diffusion_loss
                
            return loss, noisy_loss
        else:
            t_0 = tf.zeros_like(graph_indices)[:, None]
            alpha_0, sigma_0 = self.alpha_and_sigma(t_0, h)
            
            noise_h0, noise_x0 = self.sample_eps_hx_t(h, x, graph_indices)
            
            noisy_h0 = alpha_0 * h + sigma_0 * noise_h0
            noisy_x0 = alpha_0 * x + sigma_0 * noise_x0
            
            pred_h_0, pred_x_0 = self.neural(noisy_h0, noisy_x0, t_0, node_indices, pair_indices)
            reconstruction_loss_x = - self.log_px_given_z0(pred_x_0, noise_x0, graph_indices)
            reconstruction_loss_h = - self.log_ph_given_z0(h, noisy_h0, sigma_0, graph_indices) 
            reconstruction_loss = reconstruction_loss_h + reconstruction_loss_x
            # print(kl_prior_loss, diffusion_loss_t, reconstruction_loss)
            diffusion_loss = self.timesteps * diffusion_loss_t
            loss = kl_prior_loss + diffusion_loss + reconstruction_loss
            
            return loss, noisy_loss


    def forward_paras(self, s: Tensor, t: Tensor, tensor: Tensor) -> Sequence:
        """
        Compute parameters t given s in forward diffusion process.
        alpha_t_given_s = alpha_t /_alpha_s
        sigma_t_given_s_sq = -expm1(softplus(γ(s)) - softplus(γ(t)))

        Parameters
        ----------
        gamma_s : Tensor
            DESCRIPTION.
        gamma_t : Tensor
            DESCRIPTION.
        tensor : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma_t_given_s_2 = -tf.math.expm1(tf.math.softplus(gamma_s) - tf.math.softplus(gamma_t))
        sigma_t_given_s_2 = extract(sigma_t_given_s_2, tensor)
        sigma_t_given_s = tf.math.sqrt(sigma_t_given_s_2)
        
        return sigma_t_given_s, sigma_t_given_s_2
    

    def calc_mu_0(self, h_theta_0: Tensor, x_theta_0: Tensor, z0_h: Tensor, z0_x: Tensor,
                  alpha_0: Tensor, sigma_0: Tensor) -> Sequence:
        """

        Parameters
        ----------
        h_theta_0 : Tensor
            DESCRIPTION.
        x_theta_0 : Tensor
            DESCRIPTION.
        z0_h : Tensor
            DESCRIPTION.
        z0_x : Tensor
            DESCRIPTION.
        alpha_0_h : Tensor
            DESCRIPTION.
        sigma_0_h : Tensor
            DESCRIPTION.
        alpha_0_x : Tensor
            DESCRIPTION.
        sigma_0_x : Tensor
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        if self.pattern == "denoising":
            h = h_theta_0
            x = x_theta_0
        elif self.pattern == "noise":
            eps0_h = h_theta_0
            eps0_x = x_theta_0
            h = 1. / alpha_0 * (z0_h - sigma_0 * eps0_h)
            x = 1. / alpha_0 * (z0_x - sigma_0 * eps0_x)
        elif self.pattern == "score":
            score_0_h = h_theta_0
            score_0_x = x_theta_0
            h = 1. / alpha_0 * (z0_h + tf.math.square(sigma_0) * score_0_h)
            x = 1. / alpha_0 * (z0_x + tf.math.square(sigma_0) * score_0_x)
        else:
            raise ValueError(self.pattern)

        return h, x


    def calc_mu_theta_t(self, t: Tensor, s: Tensor,
                        zt_h: Tensor, zt_x: Tensor,
                        node_indices: Tensor,pair_indices: Tensor) -> Sequence:
        """

        Parameters
        ----------
        t : Tensor
            DESCRIPTION.
        s : Tensor
            DESCRIPTION.
        zt_h : Tensor
            DESCRIPTION.
        zt_x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """

        z_t = tf.concat([zt_h, zt_x], axis=-1)

        alpha_s, sigma_s = self.alpha_and_sigma(s, z_t)
        alpha_t, sigma_t = self.alpha_and_sigma(t, z_t)

        sigma_s_2 = tf.math.square(sigma_s)
        sigma_t_2 = tf.math.square(sigma_t)

        alpha_t_given_s = alpha_t / alpha_s

        sigma_t_given_s, sigma_t_given_s_2 = \
            self.forward_paras(s, t, z_t)

        h_theta_t, x_theta_t = self.neural(zt_h, zt_x, t, node_indices, pair_indices)

        if self.clip_noise:
            h_theta_t = tf.clip_by_value(h_theta_t,
                        clip_value_min=(zt_h - alpha_t) / sigma_t,
                        clip_value_max=(zt_h + alpha_t) / sigma_t,
                        )

        hx_theta_t = tf.concat([h_theta_t, x_theta_t], axis=-1)  

        if self.pattern == "denoising":
            hx_t = hx_theta_t
            mu_theta = (alpha_t_given_s * sigma_s_2 * z_t + alpha_s * sigma_t_given_s_2 * hx_t) / sigma_t_2
        elif self.pattern == "noise":
            eps_t = hx_theta_t
            mu_theta = (z_t - sigma_t_given_s_2 / sigma_t * eps_t) / alpha_t_given_s
        elif self.pattern == "score":
            score_t = h_theta_t
            mu_theta = (z_t + sigma_t_given_s_2 * score_t) / alpha_t_given_s
        else:
            raise ValueError(self.pattern)

        sigma_theta = sigma_t_given_s * sigma_s / sigma_t

        return mu_theta, sigma_theta


    def sample_hx_given_z0(self, z0_h: Tensor, z0_x: Tensor, node_indices: Tensor,
                           pair_indices: Tensor, graph_indices: Tensor) -> Sequence:
        """
        Sample hx from p(hx|z_0)

        Parameters
        ----------
        z0_h : Tensor
            DESCRIPTION.
        z0_x : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        t_0 = tf.zeros_like(graph_indices, dtype=self.dtype)[:, None]
        
        if self.builtin_schedule:
            alpha_0, sigma_0 = self.diffusion_schedule(t_0, z0_h)
        else:
            alpha_0, sigma_0 = self.alpha_and_sigma(t_0, z0_h)
        
        h_theta_0, x_theta_0 = self.neural(z0_h, z0_x, t_0, node_indices, pair_indices)
    
        mu_0_h, mu_0_x = self.calc_mu_0(h_theta_0, x_theta_0, z0_h, z0_x, alpha_0, sigma_0)
    
        eps0_h, eps0_x = self.sample_eps_hx_t(z0_h, z0_x, graph_indices)
        h = mu_0_h + sigma_0 * eps0_h
        x = mu_0_x + sigma_0 * eps0_x

        # h is scaled to 0.25*h at initial, recaling to h
        h = h / self.scaling

        # one-hot
        h = tf.one_hot(tf.argmax(h, axis=-1), self.num_class)

        return h, x


    def sample_zs_given_zt(self, t: Tensor, s: Tensor, z_t: Tensor,
                           node_indices: Tensor, pair_indices: Tensor, graph_indices: Tensor) -> Tensor:
        """

        Parameters
        ----------
        t : Tensor
            DESCRIPTION.
        s : Tensor
            DESCRIPTION.
        z_t : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        zt_h = z_t[:, : self.node_dim]
        zt_x = z_t[:, self.node_dim:]

        mu_theta, sigma_theta = self.calc_mu_theta_t(t, s, zt_h, zt_x, node_indices, pair_indices)

        noise_h, noise_x = self.sample_eps_hx_t(zt_h, zt_x, graph_indices)
        noise = tf.concat([noise_h, noise_x], axis=-1)

        z_s = mu_theta + sigma_theta * noise

        if self.re_project:
            zs_h = z_s[:, : self.node_dim]
            zs_x = z_s[:, self.node_dim:]
            zs_x = gravity_to_zero(zs_x, graph_indices)

            # assert_gravity_to_zero(zs_x, graph_indices)

            z_s = tf.concat([zs_h, zs_x], axis=-1)

        return z_s


    def sample_z1(self, graph_indices: Tensor, n_nodes: Tensor) -> Tensor:
        """

        Parameters
        ----------
        graph_indices : Tensor
            DESCRIPTION.
        n_nodes : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        z1_h = tf.random.normal(shape=(n_nodes, self.node_dim), mean=0.0, stddev=1.0)
        
        zx_gravity = tf.random.normal(shape=(n_nodes, self.x_dim), mean=0.0, stddev=1.0)   
        z1_x = gravity_to_zero(zx_gravity, graph_indices)

        # assert_gravity_to_zero(z1_x, graph_indices)
        z1 = tf.concat([z1_h, z1_x], axis=-1)
        
        return z1


    def sample_builtin(self, t: Tensor, s: Tensor, noisy_hx: Tensor,
                       node_indices: Tensor, pair_indices: Tensor, graph_indices: Tensor):
        """

        Parameters
        ----------
        t : Tensor
            DESCRIPTION.
        s : Tensor
            DESCRIPTION.
        noisy_hx : Tensor
            DESCRIPTION.
        node_indices : Tensor
            DESCRIPTION.
        pair_indices : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        noisy_hxs : TYPE
            DESCRIPTION.

        """
        
        alpha_s, sigma_s = self.diffusion_schedule(s, noisy_hx)
        alpha_t, sigma_t = self.diffusion_schedule(t, noisy_hx)

        noisy_h = noisy_hx[:, : self.node_dim]
        noisy_x = noisy_hx[:, self.node_dim:]

        pred_noise_h, pred_noise_x = self.neural(noisy_h, noisy_x, t, node_indices, pair_indices)

        if self.clip_noise:
            pred_noise_h = tf.clip_by_value(pred_noise_h,
                        clip_value_min=(noisy_h - alpha_t) / sigma_t,
                        clip_value_max=(noisy_h + alpha_t) / sigma_t,
                        )

        pred_h = (noisy_h - sigma_t * pred_noise_h) / alpha_t
        pred_x = (noisy_x - sigma_t * pred_noise_x) / alpha_t

        noisy_hs = alpha_s * pred_h + sigma_s * pred_noise_h
        noisy_xs = alpha_s * pred_x + sigma_s * pred_noise_x

        if self.re_project:
            noisy_xs = gravity_to_zero(noisy_xs, graph_indices)
            noisy_hxs = tf.concat([noisy_hs, noisy_xs], axis=-1)

        return noisy_hxs


    def sample(self, n_samples: Union[Tensor, np.array], node_indices: np.array, pair_indices: np.array) -> Sequence:
        """
        Sample final h, x by given z0

        Parameters
        ----------
        n_samples : Union[Tensor, List]
            n_samples is a sequence composed of node numbers of graphs in the batch.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """

        # assert tf.shape(n_samples)[0] == self.batch_size
        
        graph_indices = tf.repeat(tf.range(self.batch_size), n_samples)
        n_nodes = tf.math.reduce_sum(n_samples)
        
        z_s = self.sample_z1(graph_indices, n_nodes)
        
        for i in tf.range(self.timesteps, 0, delta=-1):
            i_s = tf.fill(value=i, dims=(n_nodes, 1))
            i_t = i_s + 1
            s = tf.cast(i_s / self.timesteps, dtype=self.dtype)
            t = tf.cast(i_t / self.timesteps, dtype=self.dtype)
            
            if self.builtin_schedule:
                z_s = self.sample_builtin(t, s, z_s, node_indices, pair_indices, graph_indices)
            else:
                z_s = self.sample_zs_given_zt(t, s, z_s, node_indices, pair_indices, graph_indices)
        
        z0_h = z_s[:, : self.node_dim]
        z0_x = z_s[:, self.node_dim:]
        
        # assert tf.shape(z0_x)[-1] == 3
        
        h, x = self.sample_hx_given_z0(z0_h, z0_x, node_indices, pair_indices, graph_indices)

        # partition to graphs
        h = tf.dynamic_partition(
            h, graph_indices, self.batch_size
        )

        x = tf.dynamic_partition(
            x, graph_indices, self.batch_size
        )
        
        return h, x


    def sample_diffusion(self, n_samples: Union[Tensor, np.array], node_indices: np.array, pair_indices: np.array) -> Sequence:
        """
        Sample z_t during diffusion process.

        Parameters
        ----------
        n_samples : Union[Tensor, List]
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """

        # assert tf.shape(n_samples)[0] == self.batch_size
        
        graph_indices = tf.repeat(tf.range(self.batch_size), n_samples)
        n_nodes = tf.math.reduce_sum(n_samples)
        
        z_s = self.sample_z1(graph_indices, n_nodes)
        
        # sample middle states
        for i in tf.range(self.timesteps, 0, delta=-1):
            i_s = tf.fill(value=i, dims=(n_nodes, 1))
            i_t = i_s + 1
            s = tf.cast(i_s / self.timesteps, dtype=self.dtype)
            t = tf.cast(i_t / self.timesteps, dtype=self.dtype)
            
            if self.builtin_schedule:
                z_s = self.sample_builtin(t, s, z_s, node_indices, pair_indices, graph_indices)
            else:
                z_s = self.sample_zs_given_zt(t, s, z_s, node_indices, pair_indices, graph_indices)
        
        z0_h = z_s[:, : self.node_dim]
        z0_x = z_s[:, self.node_dim:]
        
        # assert tf.shape(z0_x)[-1] == 3
        
        # sample the final state
        h, x = self.sample_hx_given_z0(z0_h, z0_x, node_indices, pair_indices, graph_indices)

        # partition to graphs
        h = tf.dynamic_partition(
            h, graph_indices, self.batch_size
        )

        x = tf.dynamic_partition(
            x, graph_indices, self.batch_size
        )
        
        return h, x


    def call(self, inputs: Sequence) -> Tensor:
        """
        Different features h and x should be scaling making them can be treated by neural network easily. 
        x: 1
        h: 0.25

        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        # assert len(inputs) == 3 or len(inputs) == 4, f"length of inputs is {len(inputs)}, should be 3 or 4!"
        
        if len(inputs) == 5:
            h, x, node_indices, pair_indices, graph_indices = inputs
        else:
            h, x, node_indices, extra, pair_indices, graph_indices = inputs
        
        # assert tf.shape(x)[-1] == self.x_dim == 3, f" the dim of x is {tf.shape(x)[-1]}, should be {self.x_dim}!"
        
        h = h * self.scaling

        x = gravity_to_zero(x, graph_indices)
        # assert_gravity_to_zero(x, graph_indices)

        loss = self.calculate_loss(h, x, node_indices, pair_indices, graph_indices)
        
        return loss
    
    
    def get_config(self):
        config = super().get_config()
        config.update({
                       "dynamics": self.dynamics,
                       "schedule": self.schedule,
                       "batch_size": self.batch_size,
                       "node_dim": self.node_dim,
                       "x_dim": self.x_dim,
                       "timesteps": self.timesteps,
                       "pattern": self.pattern,
                       "l2_loss": self.l2_loss,
                       "min_signal_rate": self.min_signal_rate,
                       "max_signal_rate": self.max_signal_rate,
                       "clip_noise": self.clip_noise,
                       "re_project": self.re_project,
                       "scaling": self.scaling,
                       "epsilon": self.epsilon,
                       "num_classes": self.num_class,
                       })
        return config