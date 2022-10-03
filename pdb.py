# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:03:41 2022

@author: huzongxiang
"""


from pathlib import Path
import pickle as pkl
import numpy as np


unit=10.0
path = Path("./")
sample_path = path/"samples_2000.pkl"
# sample_path = Path("C:\\Users\\huzon\\Desktop\\samples_2100.pkl")
save_path = path/"samples/"

residue_names = {
    0 : 'ALA', 1 : 'CYS', 2 : 'ASP', 3 : 'GLU',
    4 : 'PHE', 5 : 'GLY', 6 : 'HIS', 7 : 'LYS',
    8 : 'ILE', 9 : 'LEU', 10 : 'MET', 11 : 'ASN',
    12 : 'PRO', 13 : 'GLN', 14 : 'ARG', 15 : 'SER',
    16 : 'THR', 17 : 'VAL', 18 : 'TYR', 19 : 'TRP'}


with open(sample_path,'rb') as f:
    samples = pkl.load(f)

protein_residues = samples["h"]
protein_coords = samples["x"]

batch_size = len(protein_residues)

protein_residue_indices = []
protein_x = []
protein_y = []
protein_z = []
for i in range(batch_size):
    serials = np.argmax(protein_residues[i], -1)
    coords_x = protein_coords[i][:, 0] * unit
    coords_y = protein_coords[i][:, 1] * unit
    coords_z = protein_coords[i][:, 2] * unit
    
    protein_residue_indices.append(serials)
    protein_x.append(coords_x)
    protein_y.append(coords_y)
    protein_z.append(coords_z)
    

def write_atom_line(serial, name, resName, x, y, z, resSeq, element="C", altLoc=" ", chainID="L", iCode=" ", occupancy=1.00, tempFactor=1.00):
    if len(name) == 4:
        pdb_line = f"ATOM  {serial:>5d} {name:<4s}{altLoc}{resName:3s} {chainID}{resSeq:>4d}{iCode}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{tempFactor:>6.2f}          {element:>2s}\n"
    else:
        pdb_line = f"ATOM  {serial:>5d} {name:>2s}  {altLoc}{resName:3s} {chainID}{resSeq:>4d}{iCode}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{tempFactor:>6.2f}          {element:>2s}\n"
    return pdb_line
    
protein_pdbs = []
for i in range(batch_size):
    protein_pdb = []
    for j, residue_indice in enumerate(protein_residue_indices[i]):
        resName = residue_names[residue_indice]
        x = protein_x[i][j]
        y = protein_y[i][j]
        z = protein_z[i][j]
        name = "CA"
        atom_line = write_atom_line(serial=j, name=name, resName=resName, x=x, y=y, z=z, resSeq=j)
        protein_pdb.append(atom_line)
    protein_pdbs.append(protein_pdb)


save_path.mkdir(exist_ok=True)
for i, pdb in enumerate(protein_pdbs):
    file = "sample_" + str(i) + '.pdb'
    save_file = save_path/file
    with open(save_file, "w") as f:
        f.writelines(pdb)
    
    
    
    
    
    
    
    
    
    