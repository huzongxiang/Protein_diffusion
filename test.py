# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:28:53 2022

@author: huzongxiang
"""


import time
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from egnn import Silu
from pathlib import Path
from data import load_data
from graph import dataset, GraphBatchGenerator
from egnn import EGNNConv
from egnn_diffusion import Egnn_diffusion
from diffusion import VariationalGaussianDiffusion
from utils import (extract,
                   gravity_to_zero,
                   assert_gravity_to_zero,
                   gaussian_kl,
                   gaussian_kl_subspace,
                   standard_cdf)

from model import diffusion_model, train_model, Diffusion

# tf.device('/gpu:2')

DATASIZE = 1024
BATCH_SIZE = 128
TIMESTEPS = 2
EPOCHS = 1

data_path = Path("/data2/huzx/data/antibody/protein_respos_fixed.pkl")
datas = load_data(data_path)

pdb_ids, chains_list, residues_list, positions_list = dataset(datas[:DATASIZE])
train_data = GraphBatchGenerator(node_features_list=residues_list,
                                 node_coords_list=positions_list,
                                 batch_size=BATCH_SIZE,
                                 is_shuffle=True)

h, x, graph_indices = train_data[0][0]
t = tf.ones_like(graph_indices)
tt = extract(t, h)


num_atoms_per_graph = tf.math.bincount(graph_indices)
num_edges_per_graph = tf.math.square(num_atoms_per_graph)

@tf.function
def cal(num_atoms_per_graph):
    full_pair_indices = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    max_len = tf.shape(num_atoms_per_graph)[0]
    index = 0
    for i in tf.range(max_len):
        for j in tf.range(0, num_atoms_per_graph[i], 1):
            for k in tf.range(0, num_atoms_per_graph[i], 1):
                full_pair_indices = full_pair_indices.write(index, [j,k])
                index = index + 1
    full_pair_indices = full_pair_indices.stack()
    return full_pair_indices

def cal1(num_atoms_per_graph):
    full_pair_indices = []
    for num in num_atoms_per_graph:
        for i in tf.range(0, num, 1):
            for j in tf.range(0, num, 1):
                full_pair_indices.append([i, j])
    full_pair_indices = tf.convert_to_tensor(full_pair_indices)
    return full_pair_indices


start = time.time()
i = tf.constant(0)
result = tf.constant(0)
c = lambda i: tf.less(i, 100000)

def body(i, result):
    result = tf.sin(i)
    return i+1, result
r = tf.while_loop(c, body, [i, result])
end = time.time()
run_time = end - start
print('run time:  {:.2f} s'.format(run_time))


# start = time.time()
# full_pair_indices = cal(num_atoms_per_graph)
# end = time.time()
# run_time = end - start
# print('run time:  {:.2f} s'.format(run_time))

# start = time.time()
# full_pair_indices = cal1(num_atoms_per_graph)
# end = time.time()
# run_time = end - start
# print('run time:  {:.2f} s'.format(run_time))

# print(full_pair_indices.shape[0])
# increment = tf.cumsum(num_atoms_per_graph[:-1])
# increment = tf.pad(
#             tf.repeat(increment, num_edges_per_graph[1:]), [(num_edges_per_graph[0], 0)])

# full_pair_indices = full_pair_indices + increment[:, None]

# print(tf.config.list_physical_devices("GPU"))
# with tf.device('/gpu:0'):
#     egnn = EGNNConv()
#     egnn([h, x, full_pair_indices])

# print(tf.config.list_physical_devices("GPU"))
# with tf.device('/gpu:0'):
#     ed = Egnn_diffusion()
#     ed([h, x, tt, graph_indices])

# vgd = diffusion_model(timesteps=2)
# diffusion = Diffusion(model=vgd)
# diffusion.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
# diffusion.fit(train_data, epochs=2, batch_size=2)
