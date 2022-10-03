# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 22:03:01 2022

@author: huzongxiang
"""


import tensorflow as tf


def silu(x):
    """
    f(x)=x⋅σ(x)
    f′(x)=f(x)+σ(x)(1−f(x))

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    return x * tf.sigmoid(x) + tf.sigmoid(x) * (1 - x * tf.sigmoid(x))