# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:59:16 2022

@author: huzongxiang 
"""


import math
from tkinter import S
from typing import Sequence, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensor import Tensor
from baselayer import BaseDense
from gcn import GCNConv, GATConv, GINConv, GGNNConv, CGGCNConv
from activation import silu



class Silu(layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    
    def call(self, inputs: Tensor) -> Tensor:
        """

        Parameters
        ----------
        inputs : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        return silu(inputs)


    def get_config(self):
        config = super().get_config()
        config.update()
        return config


class Coords(layers.Layer):
    
    def __init__(self, norm_constant: float=1., **kwargs):
        super().__init__(**kwargs)
        self.norm_constant = norm_constant
        
    
    def call(self, inputs: Sequence) -> Sequence:
        """

        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.
        norm_constant : float, optional
            DESCRIPTION. The default is 1..

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        # assert len(inputs) == 2, f"length of inputs is {len(inputs)}, should be 2!"
        
        coords, edge_index = inputs
        
        coords_ = tf.gather(coords, edge_index)
        vectors = coords_[:, 0] - coords_[:, 1]
        squares = tf.math.abs(tf.math.reduce_sum(vectors ** 2, axis=-1))
        norm = tf.math.sqrt(squares + 1e-6)[:, None]

        vectors = vectors/(norm + self.norm_constant)
        return norm, vectors


    def get_config(self):
        config = super().get_config()
        config.update({
                       "norm_constant": self.norm_constant,
                       })
        return config


class EGCNConv(BaseDense):
    
    def __init__(self,
                 aggregate:str="mean",
                 hidden_dim:int=256,
                 act_fn=Silu(),
                 attention:bool=True,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation="relu",
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
        
        self.aggregate = aggregate
        self.attention = attention
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        

    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]

        self.edge_mlp =  keras.Sequential(
                                          [layers.Input(shape=(self.node_dim * 2 + self.edge_dim,)),
                                           layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),
                                           self.act_fn,
                                           layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),
                                           self.act_fn]
                                          )        

        self.node_mlp =  keras.Sequential(
                                          [layers.Input(shape=(self.node_dim + self.hidden_dim,)),
                                           layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),
                                            self.act_fn,
                                            layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),]
                                          )
        
        if self.attention:
            self.attention_mlp = layers.Dense(1,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint,
                                              use_bias=self.use_bias,
                                              bias_initializer=self.bias_initializer,
                                              bias_regularizer=self.bias_regularizer,
                                              activation="sigmoid"
                                              )
        self.built = True
                
        
    def edge_model(self, edge_attr, edge_mask=None):

        mij = self.edge_mlp(edge_attr)

        if self.attention:
            attentions = self.attention_mlp(mij)
            edge_coeff = mij * attentions
        else:
            edge_coeff = mij

        if edge_mask is not None:
            edge_coeff = edge_coeff * edge_mask
            
        return edge_coeff


    def call(self, inputs: Sequence, mask=None) -> Tensor:
        """
        
        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        edge_index : TYPE
            DESCRIPTION.
        edge_attrs : TYPE
            DESCRIPTION.
        coords : TYPE
            DESCRIPTION.
        vectors : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        coords : TYPE
            DESCRIPTION.

        """
        
        # assert len(inputs) == 5, f"length of inputs is {len(inputs)}, should be 5!"
        
        node_attrs, edge_attrs, edge_index = inputs
        
        features = tf.gather(node_attrs, edge_index)
        node_feats_cat = tf.concat([features[:, 0], features[:, 1], edge_attrs], axis=-1)
        
        edge_feats = self.edge_model(node_feats_cat)

        if self.aggregate == "sum":
            nodes_feat_aggregated = tf.math.segment_sum(edge_feats, edge_index[:, 0])
        else:
            nodes_feat_aggregated = tf.math.segment_mean(edge_feats, edge_index[:, 0])
        node_features_cat = tf.concat([node_attrs, nodes_feat_aggregated], axis=-1)
        
        node_features = self.node_mlp(node_features_cat)
        
        if mask is not None:
            node_features = node_features * mask
            
        return node_features


    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "aggregate": self.aggregate,
                       "attention": self.attention,
                       "hidden_dim": self.hidden_dim,
                       "act_fn": self.act_fn,
                       })
        return config


class EquivariantBlock(BaseDense):
    
    def __init__(self,
                 hidden_dim:int=256,
                 tanh:bool=False,
                 scope:float=10.0,
                 method:str="mean",
                 normal_factor:float=1.0,
                 act_fn=Silu(),
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation="relu",
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
        
        self.tanh = tanh
        self.scope = scope
        self.hidden_dim = hidden_dim
        self.method = method
        self.normal_factor = normal_factor
        self.act_fn = act_fn


    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]

        self.coord_mlp =  keras.Sequential(
                                           [layers.Input(shape=(self.node_dim * 2 + self.edge_dim,)),
                                            layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),
                                            self.act_fn,
                                            layers.Dense(self.hidden_dim,
                                                        kernel_initializer=self.kernel_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        use_bias=self.use_bias,
                                                        bias_initializer=self.bias_initializer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        bias_constraint=self.bias_constraint,),
                                            self.act_fn,
                                            layers.Dense(1, use_bias=False)]
                                            )
        self.built = True

    
    def call(self, inputs: Sequence, mask=None) -> Tensor:
        """
        
        Parameters
        ----------
        features : TYPE
            DESCRIPTION.
        edge_index : TYPE
            DESCRIPTION.
        edge_attrs : TYPE
            DESCRIPTION.
        coords : TYPE
            DESCRIPTION.
        vectors : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        coords : TYPE
            DESCRIPTION.

        """
        
        # assert len(inputs) == 5, f"length of inputs is {len(inputs)}, should be 5!"
        
        features, edge_attrs, coords, vectors, edge_index = inputs
        
        features = tf.gather(features, edge_index)
        
        node_feats_cat = tf.concat([features[:, 0], features[:, 1], edge_attrs], axis=-1)
        
        if self.tanh:
            vectors = vectors * tf.math.tanh(self.coord_mlp(node_feats_cat)) * self.scope
        else:
            vectors = vectors * self.coord_mlp(node_feats_cat)
        tf.compat.v1.check_numerics(vectors, "non number")
        
        if self.method == "mean":
            aggvectors = tf.math.segment_mean(vectors, edge_index[:, 0])
        elif self.method == "sum":
            aggvectors = tf.math.segment_sum(vectors, edge_index[:, 0])
        else:
            raise ValueError("aggragate method should be 'mean' or 'sum'")

        coords = coords + aggvectors / self.normal_factor
        
        if mask is not None:
            coords = coords * mask
            
        return coords

    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "tanh": self.tanh,
                       "scope": self.scope,
                       "hidden_dim": self.hidden_dim,
                       "method": self.method,
                       "normal_factor": self.normal_factor,
                       "act_fn": self.act_fn,
                       })
        return config


class SinusoidsEdgeattrs(layers.Layer):
    
    def __init__(self,
                 max_res:float=15.,
                 min_res:float=15./2000.,
                 div_factor:int=8,
                 emb_orig:bool=False,
                 **kwargs):
        super().__init__( **kwargs)
        
        
        self.max_res = max_res
        self.min_res = min_res
        self.div_factor = div_factor
        self.emb_orig = emb_orig
        
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.emb = 2 * math.pi * div_factor ** tf.cast(tf.range(self.n_frequencies), dtype=tf.float32) / max_res
    

    def call(self, inputs: Sequence) -> Tensor:
        """

        Parameters
        ----------
        norm : Tensor
            DESCRIPTION.

        Returns
        -------
        Tensor
            DESCRIPTION.

        """
        
        norm, pair_indices = inputs

        dist = tf.math.sqrt(norm + 1e-6)
        dist = dist * self.emb[None, :]
        sin_dist = tf.math.sin(dist)
        cos_dist = tf.math.cos(dist)
        dist_attrs = tf.concat([sin_dist, cos_dist], axis=-1)

        index_diff = tf.cast(tf.abs(pair_indices[:, 1] - pair_indices[:, 0]), dtype=tf.float32)[:, None]
        index_diff = index_diff * self.emb[None, :]
        sin_index = tf.math.sin(index_diff)
        cos_index = tf.math.cos(index_diff)
        index_attrs = tf.concat([sin_index, cos_index], axis=-1)

        edge_attrs = tf.concat([dist_attrs, index_attrs], axis=-1)

        edge_attrs = tf.concat([norm, dist_attrs], axis=-1) if self.emb_orig else edge_attrs
        
        return edge_attrs


    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "max_res": self.max_res,
                       "min_res": self.min_res,
                       "div_factor": self.div_factor, 
                       "emb_orig": self.emb_orig,
                       "n_frequencies": self.n_frequencies,
                       "emb": self.emb,
                       })
        return config


class EquivariantGCN(BaseDense):
    
    def __init__(self,
                 num_conv:int=1,
                 conv:str="egcn",
                 attention:bool=True,
                 steps:int=1,
                 heads:int=8,
                 stable:bool=True,
                 learning:bool=False,
                 hidden_dim:int=128,
                 act_fn=Silu(),
                 tanh:bool=False,
                 scope:float=10.0,
                 method:str="mean",
                 normal_factor:float=1.0,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 recurrent_regularizer=None,
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
        
        assert conv in ["egcn", "gcn", "gin", "gat", "ggnn", "cggcn"]
        
        self.num_conv = num_conv
        self.conv = conv
        self.attention = attention
        self.steps = steps
        self.heads = heads
        self.stable = stable
        self.learning = learning
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn
        self.tanh = tanh
        self.scope = scope
        self.method = method
        self.normal_factor = normal_factor
        
        self.coord2diff = Coords()
        
        self._layers = []
        if conv == "egcn":
            for _ in range(num_conv):
                self._layers.append(
                                    EGCNConv(
                                             hidden_dim=hidden_dim,
                                             act_fn=act_fn,
                                             attention=attention,                                
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             bias_initializer=self.bias_initializer,
                                             bias_regularizer=self.bias_regularizer,
                                             bias_constraint=self.bias_constraint,
                                             )
                                    )

        elif conv == "gcn":
            for _ in range(num_conv):
                self._layers.append(
                                    GCNConv(
                                            steps=steps,                                
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_initializer=self.bias_initializer,
                                            bias_regularizer=self.bias_regularizer,
                                            bias_constraint=self.bias_constraint,
                                            )
                                    )

        elif conv == "gin":
            for _ in range(num_conv):
                self._layers.append(
                                    GINConv(
                                            steps=steps,
                                            heads=heads,                                
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_initializer=self.bias_initializer,
                                            bias_regularizer=self.bias_regularizer,
                                            bias_constraint=self.bias_constraint,
                                            )
                                    )            

        elif conv == "gat":
            for _ in range(num_conv):
                self._layers.append(
                                    GATConv(
                                            steps=steps,
                                            heads=heads,
                                            stable=stable,
                                            learning=learning,                              
                                            kernel_initializer=self.kernel_initializer,
                                            kernel_regularizer=self.kernel_regularizer,
                                            kernel_constraint=self.kernel_constraint,
                                            bias_initializer=self.bias_initializer,
                                            bias_regularizer=self.bias_regularizer,
                                            bias_constraint=self.bias_constraint,
                                            )
                                    )  

        elif conv == "ggnn":
            for _ in range(num_conv):
                self._layers.append(
                                    GGNNConv(
                                             steps=steps,                               
                                             kernel_initializer=self.kernel_initializer,
                                             kernel_regularizer=self.kernel_regularizer,
                                             kernel_constraint=self.kernel_constraint,
                                             bias_initializer=self.bias_initializer,
                                             bias_regularizer=self.bias_regularizer,
                                             bias_constraint=self.bias_constraint,
                                             )
                                    ) 
            
        else:
            for _ in range(num_conv):
                self._layers.append(
                                    CGGCNConv(
                                              steps=steps,                           
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint,
                                              bias_initializer=self.bias_initializer,
                                              bias_regularizer=self.bias_regularizer,
                                              bias_constraint=self.bias_constraint,
                                              )
                                    ) 
            
        self.egnnblock = EquivariantBlock(
                                          hidden_dim=hidden_dim,
                                          tanh=tanh,
                                          scope=scope,
                                          method=method,
                                          normal_factor=normal_factor,
                                          act_fn=act_fn,
                                          kernel_initializer=self.kernel_initializer,
                                          kernel_regularizer=self.kernel_regularizer,
                                          kernel_constraint=self.kernel_constraint,
                                          use_bias=self.use_bias,
                                          bias_initializer=self.bias_initializer,
                                          bias_regularizer=self.bias_regularizer,
                                          bias_constraint=self.bias_constraint,
                                          )


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
        
        # assert len(inputs) == 4, f"length of inputs is {len(inputs)}, should be 4!"
        
        features, edge_attrs, coords, edge_index = inputs
        
        norm, vectors = self.coord2diff([coords, edge_index])
        
        for _layer in self._layers:
            features = _layer([features, edge_attrs, edge_index])
            tf.compat.v1.check_numerics(features, "non number")
        
        coords = self.egnnblock([features, edge_attrs, coords, vectors, edge_index], mask)
        tf.compat.v1.check_numerics(coords, "non number")
        
        if mask is not None:
            features = features * mask
        
        return features, coords


    def get_config(self):
        config = super().get_config()
        config.update({
                       "num_conv": self.num_conv,
                       "conv": self.conv,
                       "attention": self.attention,
                       "steps": self.steps, 
                       "heads": self.heads,
                       "stable": self.stable, 
                       "learning": self.learning,
                       "hidden_dim": self.hidden_dim,
                       "act_fn": self.act_fn, 
                       "tanh": self.tanh,
                       "scope": self.scope,
                       "method": self.method,
                       "normal_factor": self.normal_factor,
                       })
        return config


class EGNNConv(BaseDense):
    
    def __init__(self,
                 num_conv:int=1,
                 num_egnn:int=1,
                 conv:str="egcn",
                 attention:bool=True,
                 steps:int=1,
                 heads:int=8,
                 stable:bool=True,
                 learning:bool=False,
                 feature_dim:int=32,
                 emb_node:bool=False,
                 emb_edge:bool=False,
                 edge_dim:int=16,
                 hidden_dim:int=256,
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
                 recurrent_regularizer=None,
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
        
        assert conv in ["egcn", "gcn", "gin", "gat", "ggnn", "cggcn"]
        
        self.num_conv = num_conv
        self.num_egnn = num_egnn
        self.conv = conv
        self.attention = attention
        self.steps = steps
        self.heads = heads
        self.stable = stable
        self.learning = learning
        self.feature_dim = feature_dim
        self.emb_node = emb_node
        self.emb_edge = emb_edge
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.tanh = tanh
        self.scope = scope
        self.method = method
        self.normal_factor = normal_factor
        self.div_factor = div_factor
        self.emb_orig = emb_orig
        self.act_fn = act_fn
        
        self.coord2diff = Coords()
        self.edge_attrs = SinusoidsEdgeattrs(div_factor=div_factor, emb_orig=emb_orig)

        if self.emb_node:
            self.embed_node = layers.Dense(
                                           units=hidden_dim,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_regularizer=self.kernel_regularizer,
                                           kernel_constraint=self.kernel_constraint,
                                           use_bias=self.use_bias,
                                           bias_initializer=self.bias_initializer,
                                           bias_regularizer=self.bias_regularizer,
                                           bias_constraint=self.bias_constraint,
                                           )
        
        if self.emb_edge:
            self.embed_edge = layers.Dense(
                                           edge_dim,
                                           kernel_initializer=self.kernel_initializer,
                                           kernel_regularizer=self.kernel_regularizer,
                                           kernel_constraint=self.kernel_constraint,
                                           use_bias=self.use_bias,
                                           bias_initializer=self.bias_initializer,
                                           bias_regularizer=self.bias_regularizer,
                                           bias_constraint=self.bias_constraint,
                                           )
    
        self._layers = []
        for _ in range(num_egnn):
            self._layers.append(
                                EquivariantGCN(
                                               num_conv=num_conv,
                                               conv=conv,
                                               attention=attention,
                                               steps=steps,
                                               heads=heads,
                                               stable=stable,
                                               learning=learning,
                                               hidden_dim=hidden_dim,
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
                                )
        
        self.dense = layers.Dense(
                                  feature_dim,                              
                                  kernel_initializer=self.kernel_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  use_bias=self.use_bias,
                                  bias_initializer=self.bias_initializer,
                                  bias_regularizer=self.bias_regularizer,
                                  bias_constraint=self.bias_constraint,
                                  )


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

        # assert len(inputs) == 3, f"length of inputs is {len(inputs)}, should be 3!"
        
        features, coords, edge_index = inputs
        
        norm, _ = self.coord2diff([coords, edge_index])

        if self.emb_node:
            features = self.embed_node(features)
            
        edge_attrs = self.edge_attrs([norm, edge_index])
        if self.emb_edge:
            edge_attrs = self.embed_edge(edge_attrs)

        for _layer in self._layers:
            features, coords = _layer([features, edge_attrs, coords, edge_index])

        features = self.dense(features)

        return features, coords
    
    
    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
                       "num_conv": self.num_conv,
                       "num_egnn": self.num_egnn,
                       "conv": self.conv,
                       "attention": self.attention,
                       "steps": self.steps, 
                       "heads": self.heads,
                       "stable": self.stable, 
                       "learning": self.learning,
                       "feature_dim": self.feature_dim,
                       "emb_node": self.emb_node,
                       "emb_edge": self.emb_edge,
                       "edge_dim": self.edge_dim,
                       "hidden_dim": self.hidden_dim,
                       "tanh": self.tanh,
                       "scope": self.scope,
                       "div_factor": self.div_factor,
                       "emb_orig": self.emb_orig,
                       "act_fn": self.act_fn,
                       "method": self.method,
                       "normal_factor": self.normal_factor,
                       })
        return config