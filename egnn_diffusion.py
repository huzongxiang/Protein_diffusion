# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:51:40 2022

@author: huzongxiang
"""


import math
from typing import Sequence, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensor import Tensor
from baselayer import BaseDense
from egnn import EGNNConv, Silu


class SinusoidalPosEmb(layers.Layer):
    def __init__(self, feature_dim:int=8):
        super().__init__()
        self.feature_dim = feature_dim

    def call(self, node_indices: Tensor) -> Tensor:
        """

        Parameters
        ----------
        node_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        node_indices = tf.cast(node_indices[:, None], dtype=tf.float32)

        half_dim = int(self.feature_dim // 2)
        range_dim = tf.cast(tf.range(half_dim), dtype=tf.float32)
        emb = tf.math.log(10000.) / (half_dim - 1)
        emb = tf.math.exp(range_dim * -emb)
        emb = node_indices * emb[None, :]
        sin = tf.math.sin(emb)
        cos = tf.math.cos(emb)
        emb = tf.concat([sin, cos], axis=-1)
        
        return emb


    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "feature_dim": self.feature_dim,
                       })
        return config


class Egnn_diffusion(BaseDense):
    
    def __init__(self,
                 num_conv:int=2,
                 num_egnn:int=2,
                 conv:str="gcn",
                 timesteps:int=1000,
                 full_link:bool=False,
                 cutoff:float=16.0,
                 emb_pos:bool=True,
                 emb_t:bool=False,
                 dim_t:int=8,
                 emb_node:bool=True,
                 emb_edge:bool=False,
                 node_dim:int=20,
                 edge_dim:int=16,
                 hidden_dim:int=128,
                 steps:int=1,
                 heads:int=8,
                 stable:bool=True,
                 learning:bool=False,
                 tanh:bool=False,
                 scope:float=10.0,
                 method:str="mean",
                 normal_factor:float=1.0,
                 div_factor:int=8,
                 emb_orig:bool=False,
                 act_fn=Silu(),
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation=None,
                 **kwargs):
        super().__init__(
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         activation=activation,
                         **kwargs,
                         )
        
        # assert conv in ["gcn", "gin", "gat", "ggnn", "cggcn"]
        
        self.num_conv = num_conv
        self.num_egnn = num_egnn
        self.conv = conv
        self.timesteps = timesteps
        self.full_link = full_link
        self.cutoff = cutoff
        # self.cutoff_sq = cutoff**2
        self.emb_pos = emb_pos
        self.emb_t = emb_t
        self.dim_t = dim_t
        self.emb_node = emb_node
        self.emb_edge = emb_edge
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.heads = heads
        self.stable = stable
        self.learning = learning
        self.tanh = tanh
        self.scope = scope
        self.div_factor = div_factor
        self.emb_orig = emb_orig
        self.act_fn = act_fn
        self.method = method
        self.normal_factor = normal_factor

        if self.emb_pos:
            self.emb_position = SinusoidalPosEmb(feature_dim=self.node_dim)

        if self.emb_t:
            self.embed_time = layers.Embedding(input_dim=timesteps + 1, output_dim=dim_t)
        
        # self.embed_node = layers.Dense(units=node_dim,                             
        #                             kernel_initializer=kernel_initializer,
        #                             kernel_regularizer=kernel_regularizer,
        #                             kernel_constraint=kernel_constraint,
        #                             use_bias=use_bias,
        #                             bias_initializer=bias_initializer,
        #                             bias_regularizer=bias_regularizer,
        #                             bias_constraint=bias_constraint,
        #                             )        
        
        self.egnnconv = EGNNConv(num_conv=num_conv,
                                 num_egnn=num_egnn,
                                 conv=conv,
                                 steps=steps,
                                 heads=heads,
                                 stable=stable,
                                 learning=learning,
                                 feature_dim=node_dim,
                                 emb_node=emb_node,
                                 emb_edge=emb_edge,
                                 edge_dim=edge_dim,
                                 hidden_dim=hidden_dim,
                                 div_factor=div_factor, 
                                 emb_orig=emb_orig,
                                 act_fn=act_fn,
                                 tanh=tanh,
                                 scope=scope,
                                 method=method,
                                 normal_factor=normal_factor,
                                 kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 use_bias=self.use_bias,
                                 bias_initializer=self.bias_initializer,
                                 bias_regularizer=self.bias_regularizer,
                                 bias_constraint=self.bias_constraint,
                                 )

    
    def pair_indices(self, coords: Tensor, full_pair_indices: Tensor) -> Tensor:
        """
        Calculate the pair indices (indexes of edges), defualt set graph as full connected graph.

        Parameters
        ----------
        coords : Tensor
            DESCRIPTION.
        graph_indices : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        # tf.function 
        
        # num_atoms_per_graph = tf.math.bincount(graph_indices)
        # num_edges_per_graph = tf.math.square(num_atoms_per_graph)

        # full_pair_indices = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        # max_len = tf.shape(num_atoms_per_graph)[0]
        # index = 0
        # for i in tf.range(max_len):
        #     for j in tf.range(0, num_atoms_per_graph[i], 1):
        #         for k in tf.range(0, num_atoms_per_graph[i], 1):
        #             full_pair_indices = full_pair_indices.write(index, [j,k])
        #             index = index + 1
        # full_pair_indices = full_pair_indices.stack()
        
        # num_atoms_per_graph = tf.math.bincount(graph_indices)
        # num_edges_per_graph = tf.math.square(num_atoms_per_graph)
        
        # full_pair_indices = []
        # for num in num_atoms_per_graph:
        #     for i in tf.range(0, num, 1):
        #         for j in tf.range(0, num, 1):
        #             full_pair_indices.append([i, j])
        # full_pair_indices = tf.convert_to_tensor(full_pair_indices)
        
        # increment = tf.cumsum(num_atoms_per_graph[:-1])
        # increment = tf.pad(
        #             tf.repeat(increment, num_edges_per_graph[1:]), [(num_edges_per_graph[0], 0)])
        
        # full_pair_indices = full_pair_indices + increment[:, None]
            
        if self.full_link:
            pair_indices = full_pair_indices
        else:
            pair_coords = tf.gather(coords, full_pair_indices)
            cond = tf.reduce_sum(tf.math.squared_difference(pair_coords[:, 0], pair_coords[:, 1]), axis=-1) <= self.cutoff ** 2
            pair_indices = tf.boolean_mask(full_pair_indices, cond)
            
        return pair_indices
        
    
    def call(self, inputs: Sequence, mask=None) -> Sequence:
        """

        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.
        mask : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        # assert len(inputs) == 4 or len(inputs) == 5, f"length of inputs is {len(inputs)}, should be 4 or 5!"
        

        features, coords, t, node_indices, pair_indices = inputs
        pair_indices = self.pair_indices(coords, pair_indices)
        
        # assert tf.shape(t)[0] == tf.shape(features)[0]
        
        if self.emb_pos:
            features_pos = self.emb_position(node_indices)
            features = tf.concat([features, features_pos], axis=-1)

        if self.emb_t:
            t = tf.round(t * self.timesteps)
            embed_t = tf.squeeze(self.embed_time(t))
            features_t = tf.concat([features, embed_t], axis=-1)
        
        # assert tf.shape(t)[0] == tf.shape(features)[0]
        # features = self.embed_node(features)
        else:
            features_t = tf.concat([features, t], axis=-1)
            
        features, coords = self.egnnconv([features_t, coords, pair_indices], mask=mask)
        
        return features, coords
    
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "num_conv": self.num_conv,
                       "num_egnn": self.num_egnn,
                       "conv": self.conv, 
                       "timesteps": self.timesteps,
                       "full_link": self.full_link,
                       "cutoff": self.cutoff,
                       # "cutoff_sq": self.cutoff_sq,
                       "emb_pos": self.emb_pos,
                       "emb_t": self.emb_t,
                       "dim_t": self.dim_t,
                       "emb_node": self.emb_node,
                       "emb_edge": self.emb_edge,
                       "node_dim": self.node_dim,
                       "edge_dim": self.edge_dim,
                       "hidden_dim": self.hidden_dim,
                       "steps": self.steps,
                       "heads": self.heads,
                       "stable": self.stable,
                       "learning": self.learning,
                       "tanh": self.tanh,
                       "scope": self.scope,
                       "method": self.method,
                       "normal_factor": self.normal_factor,
                       "div_factor": self.div_factor,
                       "emb_orig": self.emb_orig,
                       "act_fn": self.act_fn,
                       })
        return config