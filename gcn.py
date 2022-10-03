# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:00:00 2022

@author: huzongxiang
"""


from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers
from baselayer import BaseConv


class GCNConv(BaseConv):
    """
    The GCN graph implementation as described in the paper
    Semi-Supervised Classification with Graph Convolutional Networks (Kipf and Welling, ICLR 2017):
    (https://arxiv.org/abs/1609.02907)
    """
    def __init__(self,
        steps:int=1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        activity_regularizer=None,
        **kwargs):
        super().__init__(
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs,
                         )
        
        self.steps = steps


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        
        with tf.name_scope("nodes_aggregate"):
            self.kernel = self.add_weight(
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias',
            )

        self.built = True
        
    
    def aggregate_nodes(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_merge_features = tf.concat([atom_features_gather[:, 0], atom_features_gather[:, 1], edge_attrs], axis=-1)

        transformed_features = tf.matmul(atom_merge_features, self.kernel) + self.bias
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:, 0])

        atom_features_updated = atom_features + atom_features_aggregated
        atom_features_updated = tf.nn.sigmoid(atom_features_updated)

        return atom_features_updated


    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs

        atom_features_updated = atom_features
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edge_attrs, pair_indices])
            
        return atom_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config


class GINConv(BaseConv):
    """
    The GIN graph implementation as described in the paper
    Gated Graph Sequence Neural Networks (Yujia Li et al. 2015):
    (http://arxiv.org/abs/1511.05493)
    """
    def __init__(self,
        epsilon:float=0.2,
        steps:int=1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        activity_regularizer=None,
        **kwargs):
        super().__init__(
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs,
                         )
        
        self.epsilon = epsilon
        self.steps = steps


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel = self.add_weight(
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias',
            )

        self.built = True
        
    
    def aggregate_nodes(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_merge_features = tf.concat([atom_features_gather[:, 0], atom_features_gather[:, 1], edge_attrs], axis=-1)

        transformed_features = tf.matmul(atom_merge_features, self.kernel) + self.bias
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:, 0])

        atom_features_updated = atom_merge_features * self.epsilon + atom_features_aggregated
        atom_features_updated = tf.nn.sigmoid(atom_features_updated)

        return atom_features_updated


    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs
        
        atom_features_updated = atom_features
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edge_attrs, pair_indices])
            
        return atom_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        config.update({"epsilon": self.epsilon})
        return config

    
class GATConv(BaseConv):
    """
    The GAT implementation as described in the paper
    Graph Attention Networks (Veličković et al., ICLR 2018):
    (https://arxiv.org/abs/1710.10903)
    """
    def __init__(self,
        steps:int=1,
        heads:int=8,
        stable:bool=True,
        learning:bool=True,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        activity_regularizer=None,
        **kwargs):
        super().__init__(
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs,
                         )
        
        self.steps = steps
        self.num_heads = heads
        self.stable = stable
        self.learning = learning


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        
        with tf.name_scope("self_attention"):
            # weight for updating atom_features
            self.kernel = self.add_weight(
                shape=(self.atom_dim, 
                       self.atom_dim * self.num_heads),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='node_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.atom_dim * self.num_heads,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='node_bias',
            )

            self.kernel_edge = self.add_weight(
                shape=(self.edge_dim, 
                       self.edge_dim * self.num_heads),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='edge_kernel',
            )
            self.bias_edge = self.add_weight(
                shape=(self.edge_dim * self.num_heads,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='edeg_bias',
            )

            if self.learning:
                self.coeff_kernel = self.add_weight(
                    shape=(self.atom_dim * 2 + self.edge_dim, 1),
                    trainable=True,
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='coeff_kernel',
                )

            self.kernel_wo = self.add_weight(
                shape=(self.num_heads * self.atom_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='proj_kernel',
            )
            self.bias_wo = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='proj_bias',
            )        

        self.leaky_relu = layers.LeakyReLU(alpha=0.2, name='leaky_relu')

        self.built = True
        
    
    def aggregate_nodes(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs

        atom_features = tf.matmul(atom_features, self.kernel) + self.bias
        edge_attrs = tf.matmul(edge_attrs, self.kernel_edge) + self.bias_edge

        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_features_receive = atom_features_gather[:, 0]
        atom_features_send = atom_features_gather[:, 1]

        atom_features_receive = tf.reshape(atom_features_receive, shape=(-1, self.num_heads, self.atom_dim))
        atom_features_send = tf.reshape(atom_features_send, shape=(-1, self.num_heads, self.atom_dim))
        edge_attrs = tf.reshape(edge_attrs, shape=(-1, self.num_heads, self.edge_dim))

        atom_merge_features = tf.concat([atom_features_receive, atom_features_send, edge_attrs], axis=-1)

        if self.stable:
            atom_merge_features = atom_merge_features - tf.math.reduce_max(atom_merge_features, axis=-1)[:, :, None]

        if self.learning:
            a = tf.matmul(atom_merge_features, self.coeff_kernel)
        else:
            a = tf.math.reduce_mean(atom_merge_features, axis=-1)
        activated_values = self.leaky_relu(a)

        exp = tf.math.exp(activated_values)
        tf.compat.v1.check_numerics(exp, "non number")        

        coeff_numerator = tf.repeat(exp, self.atom_dim, axis=-1)
        coeff_numerator = tf.reshape(coeff_numerator, shape=(-1, self.num_heads, self.atom_dim))
        coeff_denominator = tf.math.segment_sum(coeff_numerator, pair_indices[:, 0])

        atom_features_updated = tf.multiply(coeff_numerator, atom_features_send)

        atom_features_aggregated = tf.math.segment_sum(atom_features_updated, pair_indices[:, 0])
        atom_features_aggregated = tf.math.divide(atom_features_aggregated, coeff_denominator)
        atom_features_aggregated = tf.reshape(atom_features_aggregated, shape=(-1, self.num_heads * self.atom_dim))

        atom_features_aggregated = tf.matmul(atom_features_aggregated, self.kernel_wo) + self.bias_wo

        return atom_features_aggregated
    

    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs

        atom_features_updated = atom_features
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edge_attrs, pair_indices])
            
        return atom_features_updated


    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        config.update({"num_heads": self.num_heads})
        config.update({"stable": self.stable})
        config.update({"learning": self.learning})
        return config


class GGNNConv(BaseConv):
    """
    The GGNN graph implementation as described in the paper
    Gated Graph Sequence Neural Networks (Yujia Li et al. 2015):
    (http://arxiv.org/abs/1511.05493)
    """
    def __init__(self,
        steps:int=1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        activity_regularizer=None,
        **kwargs):
        super().__init__(
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs,
                         )
        
        self.steps = steps


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]

        self.update_nodes = layers.GRUCell(self.atom_dim,
                                           kernel_regularizer=self.recurrent_regularizer,
                                           recurrent_regularizer=self.recurrent_regularizer,
                                           name='update'
                                           )
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel = self.add_weight(
                shape=(self.edge_dim, 
                       self.atom_dim**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel',
            )
            self.bias = self.add_weight(
                                        shape=(self.atom_dim**2,),
                                        trainable=True,
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias',
                                        )
        
        self.built = True
       
    
    def aggregate_nodes(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        atom_features, edge_attrs, pair_indices = inputs
        
        # using bond_updated to update atom_features_neighbors
        # using num_bonds of bond feature to renew num_bonds of adjacent atom feature
        # bond feature with shape (bond_dim,), not a Matrix, multiply by a learnable weight
        # with shape (atom_dim,atom_dim,bond_dim), then bond feature transfer to shape (atom_dim,atom_dim)
        # the bond matrix with shape (atom_dim,atom_dim) can update atom_feature with shape (aotm_dim,)
        # so num_bonds of bond features need num_bonds of bond matrix, so a matrix with shape
        # (num_bonds,(atom_dim,atom_dim,bond_dim)) to transfer bond_features to shape (num_bonds,(atom_dim,atom_dim))
        # finally, apply this bond_matrix to adjacent atoms, get bond_features updated atom_features_neighbors
        edges_weights = tf.matmul(edge_attrs, self.kernel) + self.bias
        edges_weights = tf.reshape(edges_weights, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(edges_weights, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        
        # using conbination of tf.operation realizes multiplicationf between adjacent matrix and atom features
        # first tf.gather end features using end atom index pair_indices[:,1] to atom_features_neighbors
        # then using bond matrix updates atom_features_neighbors, get transformed_features
        # finally tf.segment_sum calculates sum of updated neighbors feature by start atom index pair_indices[:,0]
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:,0])

        return atom_features_aggregated


    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs
        
        atom_features_updated = atom_features

        # Perform a number of steps of message passing
        for i in range(self.steps):

            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.aggregate_nodes(
                                    [atom_features_updated, edge_attrs, pair_indices]
                                    )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_nodes(atom_features_aggregated, atom_features_updated)
            
        return atom_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config


class CGGCNConv(BaseConv):
    """
    The CGGCN graph implementation as described in the paper
    Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties (Xie et al. PRL):
    (http://arxiv.org/abs/1710.10324)
    """
    def __init__(self,
        steps:int=1,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        activity_regularizer=None,
        **kwargs):
        super().__init__(
                         activation=activation,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         **kwargs,
                         )
        
        self.steps = steps


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel_s = self.add_weight(
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_s',
            )
            self.bias_s = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_s',
            )

            self.kernel_g = self.add_weight(
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_g',
            )
            self.bias_g = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_g',
            )

        self.built = True
        
    
    def aggregate_nodes(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_merge_features = tf.concat([atom_features_gather[:, 0], atom_features_gather[:, 1], edge_attrs], axis=-1)

        transformed_features_s = tf.matmul(atom_merge_features, self.kernel_s) + self.bias_s
        transformed_features_g = tf.matmul(atom_merge_features, self.kernel_g) + self.bias_g
        
        transformed_features = tf.sigmoid(transformed_features_s) * tf.nn.softplus(transformed_features_g)
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:, 0])

        atom_features_updated = atom_features + atom_features_aggregated
        atom_features_updated = tf.nn.softplus(atom_features_updated)

        return atom_features_updated


    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        
        atom_features, edge_attrs, pair_indices = inputs

        atom_features_updated = atom_features
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edge_attrs, pair_indices])
            
        return atom_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config