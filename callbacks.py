# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:46:26 2022

@author: huzongxiang
"""


import numpy as np
import pickle as pkl
import tensorflow as tf
from pathlib import Path
from tensorflow import keras


ModulePath = Path(__file__).parent.absolute()


def full_pair_indices(num_atoms_per_graph):
    a = []
    b = []
    node_indices = []
    for num in num_atoms_per_graph:
        a.append(np.arange(0, num))
        b.append(np.concatenate([np.arange(0, num)] * num))
        node_indices.append([indice for indice in range(num)])

    node_indices = np.concatenate(node_indices, axis=0, dtype=np.float32)

    temp = np.concatenate(a, axis=0)
    reciver = np.concatenate(b, axis=0)
    
    n = np.repeat(num_atoms_per_graph, num_atoms_per_graph)
    sender = np.repeat(temp, n)
    
    full_pair_indices = np.stack([sender, reciver], axis=-1)
    
    num_edges_per_graph = np.square(num_atoms_per_graph)
    
    increment = np.cumsum(num_atoms_per_graph[:-1])
    increment = np.pad(
        np.repeat(increment, num_edges_per_graph[1:]), [(num_edges_per_graph[0], 0)])
    
    full_pair_indices = full_pair_indices + increment[:, None]
    
    return node_indices, full_pair_indices


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

       Arguments:
           patience: Number of epochs to wait after min has been hit. After this
           number of no improvement, training stops.
    """

    def __init__(self, patience=0):
        super().__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None


    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


class SavingAtMinLoss(keras.callbacks.Callback):
    """Saving training when the loss is at its min, i.e. the loss stops decreasing.

    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        Path(ModulePath/"weights").mkdir(exist_ok=True)
        self.directory = Path(ModulePath/"weights")


    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = np.Inf


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            # Record the best weights if current results is better (less).
            file = f"{self.name}-{epoch}-{self.best:8.4f}.tf"
            save_file = self.directory/file
            self.model.save_weights(save_file)


class Sampling(keras.callbacks.Callback):

    def __init__(self, n_samples, batch_size, sample_steps=100, name=None):
        super().__init__()
        self.name = name
        self.sample_steps = np.array(sample_steps)
        self.n_samples =  np.random.randint(low=n_samples[0], high=n_samples[1], size=(batch_size,))
        
        self.node_indices, self.full_pair_indices = full_pair_indices(self.n_samples)
        Path(ModulePath/"samples").mkdir(exist_ok=True)
        self.directory = Path(ModulePath/"samples")
        

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.sample_steps == 0:
            print("\nepoch: {}  sampling...".format(epoch))
            
            h, x = self.model.ema_network.layers[-1].sample(n_samples=self.n_samples,
                                                            node_indices=self.node_indices,
                                                            pair_indices=self.full_pair_indices)

            samples = {"h": h, "x": x}
            if self.name is None:
                file = f"{epoch}.pkl"
            else:
                file = f"{self.name}-{epoch}.pkl"
            save_file = self.directory/file
            with open(save_file, 'wb') as f:
                pkl.dump(samples, f, protocol=pkl.DEFAULT_PROTOCOL)
        

    def on_train_end(self, logs=None):
        print("\ntrainable: ", self.model.trainable)
        print("\nsampling after training...")
        h, x = self.model.ema_network.layers[-1].sample(n_samples=self.n_samples, pair_indices=self.full_pair_indices)
        samples = {"h": h, "x": x}
        file = f"{self.name}-f.pkl"
        save_file = self.directory/file
        with open(save_file, 'wb') as f:
            pkl.dump(samples, f, protocol=pkl.DEFAULT_PROTOCOL)