# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:01:01 2022

@author: huzongxiang
"""


from pathlib import Path
from multiprocessing import Pool
import pickle as pkl
from Bio.PDB.PDBParser import PDBParser


residue_codes = {
    'ALA' : 0, 'CYS' : 1, 'ASP' : 2, 'GLU' : 3,
    'PHE' : 4, 'GLY' : 5, 'HIS' : 6, 'LYS' : 7,
    'ILE' : 8, 'LEU' : 9, 'MET' : 10, 'ASN' : 11,
    'PRO' : 12, 'GLN' : 13, 'ARG' : 14,'SER' : 15,
    'THR' : 16, 'VAL' : 17,'TYR' : 18, 'TRP' : 19}


parser = PDBParser(PERMISSIVE=1)


def pdb_files(pdb_path):
    if not isinstance(pdb_path, Path):
        raise TypeError("path should be Path import from pathlib like Path(your_path)")
    if not pdb_path.exists():
        raise IOError("path not exists")
    pdb_files = sorted(pdb_path.glob('*.pdb'))
    return pdb_files


def pdb2respos(pdb_file):
    if not pdb_file.exists():
        raise IOError("file not exists")

    pdb_id = pdb_file.name.split('.')[0]
    structure = parser.get_structure(pdb_id, pdb_file)

    structure_dict = {}
    hetatm = {}
    disorder_residues = {}

    former_code = ""
    for chain in structure.get_chains():
        chain_dict = {}
        disorder_dict = {}
        hetatm_dict = {}
        for residue in chain.get_residues():
            residue_dict = {}
            code = residue.get_id()[1]
            icode = residue.get_id()[-1]
            resname = residue.get_resname() + "_" + str(code)
            if residue.is_disordered() < 2:
                if icode == " ":
                    if residue.get_id()[0][0] == " " and residue.get_resname()[-3:] in residue_codes:                    
                        for atom in residue:
                            if atom.name in ['CA', 'C', 'N', 'O']:
                                residue_dict[atom.name] = atom.get_coord().tolist()
                        chain_dict[resname] = residue_dict
                    else:
                        for atom in residue:
                            residue_dict[atom.name] = atom.get_coord().tolist()
                        hetatm_dict[resname] = residue_dict
                elif icode == "A":
                    if former_code != code:
                        for atom in residue:
                            if atom.name in ['CA', 'C', 'N', 'O']:
                                residue_dict[atom.name] = atom.get_coord().tolist()
                        chain_dict[resname] = residue_dict
                    else:
                        for atom in residue:
                            if atom.name in ['CA', 'C', 'N', 'O']:
                                residue_dict[atom.name] = atom.get_coord().tolist()
                        disorder_dict[resname + icode] = residue_dict
                else:
                    for atom in residue:
                        if atom.name in ['CA', 'C', 'N', 'O']:
                            residue_dict[atom.name] = atom.get_coord().tolist()
                    disorder_dict[resname + icode] = residue_dict
            else:
                for i, residue in enumerate(residue.disordered_get_list()):
                    for atom in residue:
                        if atom.name in ['CA', 'C', 'N', 'O']:
                            residue_dict[atom.name] = atom.get_coord().tolist()
                    if i == 0:
                        chain_dict[resname] = residue_dict
                    else:
                        disorder_dict[resname] = residue_dict
            former_code = residue.get_id()[1]
        structure_dict[chain.get_id()] = chain_dict
        disorder_residues[chain.get_id()] = disorder_dict
        hetatm[chain.get_id()] = hetatm_dict

    return {pdb_id : {"residues" : structure_dict, "disorder_residues": disorder_residues, "hetatm" : hetatm}}


def get_respos(pdbs):
    pool = Pool()
    res_pos = pool.map(pdb2respos, pdbs)
    pool.close()
    pool.join()
    return res_pos


def to_pkl(res_pos, save_path=None):
    if save_path is None:
        raise IOError("save_path should be given")
    pkl_file = Path(save_path/"protein_respos.pkl")
    with open(pkl_file,'wb') as f:
        pkl.dump(res_pos, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_data(path):
    if not isinstance(path, Path):
        raise TypeError("path should be Path import from pathlib like Path(your_path)")
    if not path.exists():
        raise IOError("path not exists")
    
    print("read datas...")
    with open(path,'rb') as f:
        datas = pkl.load(f)
    print("done")
        
    return datas

