# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:33:14 2022

@author: huzongxiang
"""


from typing import Sequence, Dict
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints


class BaseConv(layers.Layer):
    """
    base layer for graph convolution
    config for custom weight and bias
    """
    def __init__(self,
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
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


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

        raise NotImplementedError


    def call(self, inputs: Sequence):
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
        
        raise NotImplementedError


    def get_config(self) -> Dict:
        """
        Part of keras layer interface, where the signature is converted into a dict
        Returns:
            configurational dictionary

        Returns
        -------
        Dict
            DESCRIPTION.

        """
        
        config = {
                  "activation": activations.serialize(self.activation),
                  "use_bias": self.use_bias,
                  "kernel_initializer": initializers.serialize(self.kernel_initializer),
                  "bias_initializer": initializers.serialize(self.bias_initializer),
                  "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                  "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                  "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                  "kernel_constraint": constraints.serialize(self.kernel_constraint),
                  "bias_constraint": constraints.serialize(self.bias_constraint),
                  }

        config.update(config)
        return config


class BaseDense(layers.Layer):
    """
    base layer for graph convolution
    config for dense layer
    """
    def __init__(self,
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
        super().__init__(**kwargs)

        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.activation = activation
        

    def call(self, inputs: Sequence):
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
        
        raise NotImplementedError


    def get_config(self) -> Dict:
        """
        Part of keras layer interface, where the signature is converted into a dict
        Returns:
            configurational dictionary

        Returns
        -------
        Dict
            DESCRIPTION.

        """
        
        config = {
                  "use_bias" : self.use_bias,
                  "kernel_initializer": self.kernel_initializer,
                  "bias_initializer": self.bias_initializer,
                  "kernel_regularizer": self.kernel_regularizer,
                  "bias_regularizer": self.bias_regularizer,
                  "kernel_constraint": self.kernel_constraint,
                  "bias_constraint": self.bias_constraint,
                  "activation": self.activation,
                  }

        config.update(config)
        return config