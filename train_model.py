# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:59:58 2022

@author: huzongxiang
"""


from re import T
import warnings
import logging
from pathlib import Path
from data import load_data
import numpy as np
import tensorflow as tf
from tensorflow import keras
from augmented_data import augmented_data
from graph import dataset, GraphBatchGenerator
from model import diffusion_model, train_model, Diffusion
from callbacks import EarlyStoppingAtMinLoss, SavingAtMinLoss, Sampling
import matplotlib.pyplot as plt


tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


DATASIZE = 8
BATCH_SIZE = 2
TIMESTEPS = 20
EPOCHS = 2
SAMPLING = True
SAMPLING_STEPS = 500

weights = None # "./weight-egcn-2048-5000-nanometer-32-2e-05.tf"

schedule="learned"                      # noise schedule, should be "builtin", "learned" and "predefined"
unit="nanometer"                        # position unit "20/10nanometer", "nanometer" or "angstrom", nanometer = angstrom / 10.0
augmented=0                             # augmented data, if training data is not enough, augmented data should be True (int: > 0)
num_conv=2                              # num of stacked convolution layers in single egnn layer (unshared weights)
num_egnn=2                              # num of stacked egnn layers (unshared weights)
conv="egcn"                             # convolution
lr=1e-3                                 # learning rate
full_link=True                          # full connected graph
cutoff=16.0                             # threshold define the edges between nodes when full_link=False
scaling=1.00                            # scaling between node feature and node coordinations
tanh=False                              # tanh function inplemented in equivariant network
scope=10.0                              # scope when update node coordinations when tanh=True
steps=1                                 # steps of convolution (shared weights)
hidden_dim=64                           # hidden_dim in convolution
clip_noise=True                         # clip noise of neural during sampling
stable=True                             # stablize GAT
learning=False                          # learning weight in GAT
decay=0.995                             # decay for ema, usually be 0.999 or 0.995

#########################################################################################################################

name = f"{conv}-{BATCH_SIZE}-{TIMESTEPS}-{EPOCHS}-{unit}-{num_conv}-{num_egnn}-{hidden_dim}-{lr}"

print("\nparameters ",
      "\nbatch size: ", BATCH_SIZE,
      "\ntimes step: ", TIMESTEPS,
      "\nepoch: ", EPOCHS,
      "\nsampling: ", SAMPLING,
      "\nsamling_steps: ", SAMPLING_STEPS,
      "\nschedule: ", schedule,
      "\nunit: ", unit,
      "\naugmented: ", augmented,
      "\nnum_conv: ", num_conv,
      "\nnum_egnn: ", num_egnn,
      "\nconvolution: ", conv,
      "\nlearning rate: ", lr,
      "\nfull link: ", full_link,
      "\ncutoff: ", cutoff,
      "\nscaling: ", scaling,
      "\ntanh: ", tanh,
      "\nscope: ", scope,
      "\nsteps: ", steps,
      "\nhidden_dim: ", hidden_dim,
      "\nclip_noise: ", clip_noise,
      "\nstable: ", stable,
      "\nlearning: ", learning,
      "\ndecay: ", decay,
      )

##############################################################################################################

data_path = Path("C:\\Users\\huzon\\Desktop\\protein_respos_fixed.pkl")
# data_path = Path("/data2/huzx/data/antibody/protein_respos.pkl")
datas = load_data(data_path)

pdb_ids, chains_list, residues_list, positions_list = dataset(datas[:DATASIZE], mode="single", unit=unit)
split = round(0.8 * len(residues_list))

if augmented:
    augmented_residues_list, augmented_positions_list = augmented_data(residues_list, positions_list, augmented)
    residues_list.extend(augmented_residues_list)
    positions_list.extend(augmented_positions_list)

train_data = GraphBatchGenerator(node_features_list=residues_list[:split],
                                 node_coords_list=positions_list[:split],
                                 batch_size=BATCH_SIZE,
                                 is_shuffle=True)

valid_data = GraphBatchGenerator(node_features_list=residues_list[split:],
                                 node_coords_list=positions_list[split:],
                                 batch_size=BATCH_SIZE,
                                 is_shuffle=True)

## customized train_step
# vgd = diffusion_model(batch_size=BATCH_SIZE, timesteps=TIMESTEPS)
# train_model(model=vgd, train_data=train_data, epochs=EPOCHS)

# keras model built-in train_step
vgd = diffusion_model(schedule=schedule,
                      num_conv=num_conv,
                      num_egnn=num_egnn,
                      conv=conv,
                      batch_size=BATCH_SIZE,
                      timesteps=TIMESTEPS,
                      full_link=full_link,
                      cutoff=cutoff,
                      scaling=scaling,
                      tanh=tanh,
                      steps=steps,                              
                      hidden_dim=hidden_dim,
                      clip_noise=clip_noise,
                      stable=stable,
                      learning=learning,                  
                      )

diffusion = Diffusion(model=vgd, decay=decay)

if weights is not None:
    print("load model...")
    diffusion.load_weights(weights)

print("complie...")
# initial_learning_rate = lr
# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate, decay_steps=500, decay_rate=0.90, staircase=True
# )

# clipnorm=1.0
diffusion.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0))

print("training...")
sampling = None
saving = SavingAtMinLoss(name=name)
callbacks = [saving]
if SAMPLING:
    sampling = Sampling(n_samples=(90, 110), batch_size=BATCH_SIZE, sample_steps=SAMPLING_STEPS, name=name)
    stopping = EarlyStoppingAtMinLoss()
    callbacks=[sampling, saving]
history = diffusion.fit(train_data, validation_data=valid_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

print("saving model...")
diffusion.save_weights(f"./weight-{name}.tf")

##############################################################################################################

print("plot...")
plt.plot(history.history["loss"], label="train loss")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.legend(fontsize=16)
png_path = name + ".png"
plt.savefig(png_path)

print("\nparameters ",
      "\nbatch size: ", BATCH_SIZE,
      "\ntimes step: ", TIMESTEPS,
      "\nepoch: ", EPOCHS,
      "\nsampling: ", SAMPLING,
      "\nsamling_steps: ", SAMPLING_STEPS,
      "\nschedule: ", schedule,
      "\nunit: ", unit,
      "\naugmented: ", augmented,
      "\nnum_conv: ", num_conv,
      "\nnum_egnn: ", num_egnn,
      "\nconvolution: ", conv,
      "\nlearning rate: ", lr,
      "\nfull link: ", full_link,
      "\ncutoff: ", cutoff,
      "\nscaling: ", scaling,
      "\ntanh: ", tanh,
      "\nscope: ", scope,
      "\nsteps: ", steps,
      "\nhidden_dim: ", hidden_dim,
      "\nclip_noise: ", clip_noise,
      "\nstable: ", stable,
      "\nlearning: ", learning,
      "\ndecay: ", decay,
      )