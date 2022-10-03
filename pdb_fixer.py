# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:01:01 2022

@author: huzongxiang
"""


import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from Bio.PDB.PDBParser import PDBParser
from pdbfixer import PDBFixer
from simtk.openmm import Vec3
from simtk.openmm.app.pdbfile import PDBFile

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def pdb_files(pdb_path):
    if not pdb_path.exists():
        raise IOError("path not exists")

    pdb_files = sorted(pdb_path.glob('*.pdb'))
    return pdb_files


def fix_dl(pdbs):
    pool = Pool()
    fixed = pool.map(fix_pdb_dl, pdbs)
    pool.close()
    pool.join()


def fix(pdbs):
    pool = Pool()
    fixed = pool.map(fix_pdb, pdbs)
    pool.close()
    pool.join()


def fix_pdb(pdb_file, save_path=None):
    if not isinstance(pdb_file, Path):
        raise TypeError("pdb_file should be Path import from pathlib")
    if not pdb_file.exists():
        raise IOError("file not exists")
    if save_path is None:
        save_path = Path(pdb_file.parent.parent/"fixed")
        save_path.mkdir(exist_ok=True)
        
    if pdb_file.is_file():
        print("Creating PDBFixer...")
        fixer = PDBFixer(pdb_file.as_posix())
        print("Finding missing residues...")
        fixer.findMissingResidues()

        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                print("fixed missing...")
                del fixer.missingResidues[key]

        print("Finding nonstandard residues...")
        fixer.findNonstandardResidues()
        print("Replacing nonstandard residues...")
        fixer.replaceNonstandardResidues()
        print("Removing heterogens...")
        fixer.removeHeterogens(keepWater=True)

        print("Finding missing atoms...")
        fixer.findMissingAtoms()
        print("Adding missing atoms...")
        fixer.addMissingAtoms()
        print("Adding missing hydrogens...")
        fixer.addMissingHydrogens(7)
        
        print("Adding water box...")
        maxSize = max(max((pos[i] for pos in fixer.positions))-min((pos[i] for pos in fixer.positions)) for i in range(3))
        boxSize = maxSize*Vec3(1, 1, 1)
        fixer.addSolvent(boxSize)

        print("Writing PDB file...")
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(save_path.joinpath("%s_fixed_pH%s.pdb" % (pdb_file.name.split('.')[0], 7)),
                 "w"),
            keepIds=True)
        print("done...\n")
        del fixer
        return "%s_fixed_pH%s.pdb" % (pdb_file.name.split('.')[0], 7)


def fix_pdb_dl(pdb_file, save_path=None):
    if not isinstance(pdb_file, Path):
        raise TypeError("pdb_file should be Path import from pathlib")
    if not pdb_file.exists():
        raise IOError("file not exists")
    if save_path is None:
        save_path = Path(pdb_file.parent.parent/"fixed")
        save_path.mkdir(exist_ok=True)
        
    if pdb_file.is_file():
        print("Creating PDBFixer...")
        fixer = PDBFixer(pdb_file.as_posix())
        print("Finding missing residues...")
        fixer.findMissingResidues()

        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                print("fixed missing...")
                del fixer.missingResidues[key]

        # print("Finding nonstandard residues...")
        # fixer.findNonstandardResidues()
        # print("Replacing nonstandard residues...")
        # fixer.replaceNonstandardResidues()
        print("Removing heterogens...")
        fixer.removeHeterogens(keepWater=False)

        print("Finding missing atoms...")
        fixer.findMissingAtoms()
        print("Adding missing atoms...")
        fixer.addMissingAtoms()

        print("Writing PDB file...\n")
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(save_path.joinpath("%s_fixed.pdb" % (pdb_file.name.split('.')[0])),
                 "w"),
            keepIds=True)
        print("done...\n")
        return "%s_fixed.pdb" % (pdb_file.name.split('.')[0])


if __name__ == '__main__':
    # pdb_file = Path("./1A0Q_1.pdb")
    # save_path = pdb_file.parent
    # fix_pdb(pdb_file, save_path)
    # fix_pdb_dl(pdb_file, save_path)

    path = Path("./samples")
    save_path = path.parent
    pdbs = pdb_files(path)
    fix_dl(pdbs)