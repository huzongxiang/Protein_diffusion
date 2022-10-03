# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:26:06 2022

@author: huzongxiang
"""


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    # register variables of the trained model
    def register(self):
        for param in self.model.variables:
            if param.trainable:
                self.shadow[param.name] = param.value()

    # update variables of ema model 
    def update(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                new_average = (1.0 - self.decay) * param.value() + self.decay * self.shadow[param.name]
                self.shadow[param.name] = new_average

    # apply ema variables to training model, backup the trained model
    def apply_shadow(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.shadow
                self.backup[param.name] = param.value()
                param.assign(self.shadow[param.name])

    # restore model weights
    def restore(self):
        for param in self.model.variables:
            if param.trainable:
                assert param.name in self.backup
                param.assign(self.backup[param.name])
        self.backup = {}