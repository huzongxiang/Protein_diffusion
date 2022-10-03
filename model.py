# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 18:18:45 2022

@author: huzongxiang
"""


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from egnn import Silu
from ema import ExponentialMovingAverage
from egnn_diffusion import Egnn_diffusion
from diffusion import VariationalGaussianDiffusion
from utils import (extract,
                   gravity_to_zero,
                   assert_gravity_to_zero,
                   gaussian_kl,
                   gaussian_kl_subspace,
                   standard_cdf)


def diffusion_model(schedule:str="builtin",
                    batch_size:int=8,
                    node_dim:int=20,
                    x_dim:int=3,
                    emb_t:bool=False,
                    dim_t:int=8,
                    timesteps:int=1024,
                    pattern:str="noise",
                    l2_loss:bool=True,
                    clip_noise:bool=True,
                    re_project:bool=True,
                    scaling:float=0.25,
                    num_conv:int=2,
                    num_egnn:int=2,
                    conv:str="gcn",
                    full_link:bool=False,
                    cutoff:float=20.0,
                    steps:int=1,
                    heads:int=8,
                    stable:bool=True,
                    learning:bool=False,
                    emb_pos:bool=True,
                    emb_node:bool=True,
                    emb_edge:bool=False,
                    edge_dim:int=16,
                    hidden_dim:int=128,
                    tanh:bool=False,
                    scope:float=10.0,
                    method:str="mean",
                    normal_factor:float=1.0,
                    div_factor:int=10,
                    emb_orig:bool=False,
                    act_fn=Silu(),
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=None,
                    kernel_constraint=None,
                    use_bias=True,
                    bias_initializer="zeros",
                    bias_regularizer=None,
                    bias_constraint=None,
                    ):
    
    input_node = keras.Input(shape=(node_dim), dtype=tf.float32)
    input_coord = keras.Input(shape=(x_dim), dtype=tf.float32)
    # input_t = keras.Input(shape=(1), dtype=tf.float32)
    input_indice = keras.Input(shape=(), dtype=tf.int32)
    input_index = keras.Input(shape=(2), dtype=tf.int32)
    input_graph = keras.Input(shape=(), dtype=tf.int32)
    
    dynamics = Egnn_diffusion(num_conv=num_conv,
                              num_egnn=num_egnn,
                              conv=conv,
                              timesteps=timesteps,
                              full_link=full_link,
                              cutoff=cutoff,
                              emb_pos=emb_pos,
                              emb_t=emb_t,
                              dim_t=dim_t,
                              steps=steps,
                              heads=heads,
                              stable=stable,
                              learning=learning,
                              emb_node=emb_node,
                              emb_edge=emb_edge,
                              node_dim=node_dim,
                              edge_dim=edge_dim,
                              hidden_dim=hidden_dim,
                              div_factor=div_factor,
                              emb_orig=emb_orig,
                              act_fn=act_fn,
                              tanh=tanh,
                              scope=scope,
                              method=method,
                              normal_factor=normal_factor,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer,
                              kernel_constraint=kernel_constraint,
                              use_bias=use_bias,
                              bias_initializer=bias_initializer,
                              bias_regularizer=bias_regularizer,
                              bias_constraint=bias_constraint,
                              )
    
    # dynamics([input_node, input_coord, input_t, input_index])
    
    vgd = VariationalGaussianDiffusion(dynamics=dynamics,
                                       schedule=schedule,
                                       batch_size=batch_size,
                                       node_dim=node_dim,
                                       x_dim=x_dim,
                                       timesteps=timesteps,
                                       pattern=pattern,
                                       l2_loss=l2_loss,
                                       clip_noise=clip_noise,
                                       re_project=re_project,
                                       scaling=scaling,
                                       )
    
    x = vgd([input_node, input_coord, input_indice, input_index, input_graph])
    
    return keras.Model([input_node, input_coord, input_indice, input_index, input_graph], x, name="model")


def train_model(model, train_data, valid_data=None, epochs=10, lr=1e-4):


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    @tf.function
    def train_step(inputs):

        with tf.GradientTape() as tape:
            loss = model(inputs)
            loss = tf.math.reduce_mean(loss)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        return loss


    def valid_step(inputs):

        loss = model(inputs)
        tf.math.reduce_mean(loss)
        
        return loss


    logs = 'Epoch={}, Loss:{}'


    for epoch in range(epochs):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        # dist_iterator = iter(train_dist_dataset)
        for x in train_data:
            total_loss += train_step(x[0])
            num_batches += 1
        train_loss = total_loss / num_batches

        if valid_data is not None:
        # VALID LOOP
            for x in valid_data:
                valid_step(x)
        
        if epoch%1 == 0:
            tf.print(tf.strings.format(logs, (epoch, train_loss)))
            tf.print("")


def parallel_train(model, train_data, valid_data=None, batch_size=1, epochs=10, lr=1e-3):

    # Distribute strategy
    strategy = tf.distribute.MirroredStrategy()

    batch_size_per_replica = batch_size

    # Global batch size
    GLOBAL_BATCH_SIZE = batch_size_per_replica * strategy.num_replicas_in_sync

    # Buffer size for data loader
    BUFFER_SIZE = batch_size_per_replica * strategy.num_replicas_in_sync * 16

    # distribute dataset
    train_dist_dataset = strategy.experimental_distribute_dataset(train_data)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_data)

    # strategy
    with strategy.scope():

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

        def train_step(inputs):

            with tf.GradientTape() as tape:
                loss = model(inputs)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            return loss


        def valid_step(inputs):

            loss = model(inputs)

        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(
                train_step, args=(dataset_inputs,)
            )
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def distributed_valid_step(dataset_inputs):
            return strategy.run(valid_step, args=(dataset_inputs,))

        for epoch in range(epochs):
            # TRAIN LOOP
            total_loss = 0.0
            num_batches = 0
            # dist_iterator = iter(train_dist_dataset)
            for x in train_dist_dataset:
                total_loss += distributed_train_step(x)
                num_batches += 1
            train_loss = total_loss / num_batches

            if valid_data is not None:
            # VALID LOOP
                for x in valid_dist_dataset:
                    distributed_valid_step(x)
    

class Diffusion(keras.Model):
    """"  trainning model of Diffusin """

    def __init__(self, model, decay=0.999, **kwargs):
        super().__init__(**kwargs)
        self.network = model
        # self.ema = tf.train.ExponentialMovingAverage(decay=decay)

        # self.ema = ExponentialMovingAverage(model=self.network, decay=decay)
        # self.ema.register()

        self.ema_network = keras.models.clone_model(self.network)
        self.ema = decay
        
        self.noise_loss_tracker = keras.metrics.Mean(name="loss")
        self.noisy_loss_tracker = keras.metrics.Mean(name="f_loss")


    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.noisy_loss_tracker]


    def loss_fn(self, inputs, trainable=True):
        if trainable:
            noise_loss, noisy_loss = self.network(inputs)
        else:
            noise_loss, noisy_loss = self.ema_network(inputs)
        return tf.math.reduce_mean(noise_loss), tf.math.reduce_mean(noisy_loss)
    

    def train_step(self, data):
        with tf.GradientTape() as tape:
            noise_loss, noisy_loss = self.loss_fn(data[0])

        tf.compat.v1.check_numerics(noise_loss, "non number")
        tf.compat.v1.check_numerics(noisy_loss, "non number")
        
        grads = tape.gradient(noise_loss, self.weights)
        for grad in grads:
            if grad is not None:
                tf.compat.v1.check_numerics(grad, "non number")
        for weight in self.weights:
            tf.compat.v1.check_numerics(weight, "non number")  

        # ema 
        # opt_op = self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # with tf.control_dependencies([opt_op]):
        #     self.ema.apply(self.trainable_variables)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        # self.ema.update()

        self.noise_loss_tracker.update_state(noise_loss)
        self.noisy_loss_tracker.update_state(noisy_loss)

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        noise_loss, noisy_loss = self.loss_fn(data[0], trainable=False)
        tf.compat.v1.check_numerics(noise_loss, "non number")
        tf.compat.v1.check_numerics(noisy_loss, "non number")
        
        self.noise_loss_tracker.update_state(noise_loss)
        self.noisy_loss_tracker.update_state(noisy_loss)

        return {m.name: m.result() for m in self.metrics}