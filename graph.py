# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:01:01 2022

@author: huzongxiang
"""


import time
import json
import numpy as np
from pathlib import Path
from operator import itemgetter
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import Union, Dict, List, Set


residue_codes = {
    'ALA' : 0, 'CYS' : 1, 'ASP' : 2, 'GLU' : 3,
    'PHE' : 4, 'GLY' : 5, 'HIS' : 6, 'LYS' : 7,
    'ILE' : 8, 'LEU' : 9, 'MET' : 10, 'ASN' : 11,
    'PRO' : 12, 'GLN' : 13, 'ARG' : 14, 'SER' : 15,
    'THR' : 16, 'VAL' : 17, 'TYR' : 18, 'TRP' : 19}


def convert_to_one_hot(x, k):
    return np.eye(k, dtype=np.int32)[x]


def get_residues_position(input_dict, mode="single"):
    assert mode in ["single", "multi"], f"{mode} should be single or multi chain(s)"

    pdb_id = list(input_dict.keys())[0]
    protein_dict = input_dict[pdb_id]
    protein = protein_dict["residues"]

    chain_ids = list(protein.keys())
    first_chain = chain_ids[0]

    residues = []
    positions = []
    if mode == "single":
            res_pos = protein[first_chain]
            for k,v in res_pos.items():
                residues.append(residue_codes[k.split("_")[0]])
                positions.append(list(v.values()))
            assert len(residues) == len(positions), f"lenght of residues not equel to its position"
            return pdb_id, first_chain, residues, positions
    else:
        for chain in protein.values():
            for k,v in chain.items():
                residues.append(residue_codes[k.split("_")[0]])
                positions.append(list(v.values()))
            assert len(residues) == len(positions), f"lenght of residues not equel to its position"
        return pdb_id, chain_ids, residues, positions


def get_residues_alpha_C(input_dict, mode="single"):
    assert mode in ["single", "multi"], f"{mode} should be 'single' or 'multi' chain(s)"

    pdb_id = list(input_dict.keys())[0]
    # print(pdb_id)
    protein_dict = input_dict[pdb_id]
    protein = protein_dict["residues"]

    chain_ids = list(protein.keys())
    first_chain = chain_ids[0]

    residues = []
    alpha_C = []
    if mode == "single":
        res_pos = protein[first_chain]
        for k,v in res_pos.items():
            residues.append(residue_codes[k.split("_")[0]])
            alpha_C.append(v["CA"])
        assert len(residues) == len(alpha_C), f"lenght of residues not equel to its position"
        return pdb_id, first_chain, residues, alpha_C
    else:
        for chain in protein.values():
            for k,v in chain.items():
                residues.append(residue_codes[k.split("_")[0]])
                alpha_C.append(v["CA"])
                assert len(residues) == len(alpha_C), f"lenght of residues not equel to its position"
        return pdb_id, chain_ids, residues, alpha_C


def dataset(datas, mode="single", cutoff=500, unit="nanometer"):
    assert mode in ["single", "multi"], f"{mode} should be 'single' or 'multi' chain(s)"
    assert unit in ["20nanometer", "10nanometer", "nanometer", "angstrom"],\
    f"{unit} should be '20nanometer', '10nanometer', 'nanometer' or 'angstrom'"

    if unit == "20nanometer":
        denom = 200.0
    elif unit == "10nanometer":
        denom = 100.0
    elif unit == "nanometer":
        denom = 10.0
    elif unit == "angstrom":
        denom = 1.0
    else:
        raise ValueError("not supported unit.")

    pdb_ids = []
    chains_list = []
    residues_list = []
    positions_list = []
    for i, data in enumerate(datas):
        try:
            pdb_id, chains, residues, positions = get_residues_alpha_C(input_dict=data, mode=mode)
            if mode == "single":
                pdb_ids.append(pdb_id)
                chains_list.append(chains)
                residues_list.append(convert_to_one_hot(np.array(residues), 20))
                positions_list.append(np.array(positions, dtype=np.float64) / denom)
            else:
                if len(residues) <= cutoff:
                    pdb_ids.append(pdb_id)
                    chains_list.append(chains)
                    residues_list.append(convert_to_one_hot(np.array(residues), 20))
                    positions_list.append(np.array(positions, dtype=np.float64) / denom)
        except:
            print(i)
    return pdb_ids, chains_list, residues_list, positions_list


class  GraphBatchGenerator(Sequence):   
    
    def __init__(self,
                node_features_list: List[np.ndarray],
                node_coords_list: List[np.ndarray],
                labels: Union[List, None]=None,
                batch_size: int=32,
                is_shuffle: bool=False):
        """
        Parameters
        ----------
        X_tensor : TYPE
            DESCRIPTION.
        y_label : TYPE
            DESCRIPTION.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 32.
        is_shuffle : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """

        self.data_size = len(node_features_list)
        self.batch_size = batch_size
        self.total_index = np.arange(self.data_size)

        self.node_features_list = node_features_list
        self.node_coords_list = node_coords_list
        self.labels = labels

        if is_shuffle:
            shuffle = itemgetter(np.random.permutation(self.total_index))
            self.total_index = shuffle(self.total_index)
    
    
    def __len__(self) -> int:
        return int(np.ceil(self.data_size / self.batch_size))


    def on_epoch_end(self):
        """
        code to be executed on epoch end
        """
        self.total_index = np.random.permutation(self.total_index)


    def __getitem__(self, index: int) -> tuple:
        batch_index = self.total_index[index * self.batch_size : (index + 1) * self.batch_size]
        get = itemgetter(*batch_index)

        node_features_list = get(self.node_features_list)
        node_coords_list = get(self.node_coords_list)


        inputs_batch = (node_features_list, node_coords_list)

        x_batch = self._merge_batch(inputs_batch)
        if self.labels is None:
            return (x_batch, )
        y_batch = np.array(get(self.labels))

        return x_batch, (y_batch)


    def _merge_batch(self, x_batch: tuple) -> tuple:
        """
        Merging a batch of graphs into a disconnected graph should reindex atoms only
        features of graphs desn't be changed only merge them to one dimension of globl graph.
        reindex indices in pair_indices by adding increment of number of atoms in the batch
        atom marked with structure indice also need to be tell in globl graph.
        Parameters
        ----------
        x_batch : TYPE
            DESCRIPTION.

        Returns
        -------
        node_features : TYPE
            DESCRIPTION.
        bond_features : TYPE
            DESCRIPTION.
        state_attributes : TYPE
            DESCRIPTION.
        pair_indices : TYPE
            DESCRIPTION.
        atom_partition_indices: TYPE
            DESCRIPTION.
        bond_partition_indices: TYPE
            DESCRIPTION.
        """
        node_features, node_coords = x_batch
    
        # Obtain number of atoms and bonds for each graph
        # allocate graph (structure) indice for atom in global graph
        num_atoms_per_graph = []
        graph_indices = []
        node_indices = []
        a = []
        b = []
        for i, atoms in enumerate(node_features):
            num = len(atoms)
            num_atoms_per_graph.append(num)
            graph_indices += [i] * num
            node_indices.append([indice for indice in range(num)])
            a.append(np.arange(0, num))
            b.append(np.concatenate([np.arange(0, num)] * num))
            
        graph_indices = np.array(graph_indices, dtype=np.int32)
        node_indices = np.concatenate(node_indices, axis=0, dtype=np.int32)
        node_features = np.concatenate(node_features, axis=0, dtype=np.float32)
        node_coords = np.concatenate(node_coords, axis=0, dtype=np.float32)

        num_edges_per_graph = np.square(num_atoms_per_graph)

        temp = np.concatenate(a, axis=0)
        reciver = np.concatenate(b, axis=0)
        
        n = np.repeat(num_atoms_per_graph, num_atoms_per_graph)
        sender = np.repeat(temp, n)
        
        full_pair_indices = np.stack([sender, reciver], axis=-1)
        
        # full_pair_indices = []
        # for num in num_atoms_per_graph:
        #     for i in range(0, num, 1):
        #         for j in range(0, num, 1):
        #             full_pair_indices.append([i, j])
        # full_pair_indices = np.array(full_pair_indices)
        
        increment = np.cumsum(num_atoms_per_graph[:-1])
        increment = np.pad(
            np.repeat(increment, num_edges_per_graph[1:]), [(num_edges_per_graph[0], 0)])
        
        full_pair_indices = full_pair_indices + increment[:, None]

        return (node_features, node_coords, node_indices, full_pair_indices, graph_indices)