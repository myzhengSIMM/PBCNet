import multiprocessing
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem
import pickle
from rdkit import Chem
from collections import defaultdict
import copy
import os
import glob
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
import pickle

import torch
import dgl

from scipy.spatial import distance_matrix

from typing import List, Tuple, Union, Any, Dict, List, Tuple, Union
from itertools import zip_longest
import logging

import warnings

def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)



warnings.filterwarnings('ignore')


# =========== 用于识别蛋白和配体分子的类型，用于计算距离衰减 ===========
ligand_atom_type = {"C_3": 1, 'C_2': 2, 'C_1': 3, 'C_ar': 4, 'O_3': 5, 'O_3p': 6, 'O_2': 7, 'O_co2': 8, 'O_2v': 9,
                    'N_2': 10, 'N_am': 11, 'N_pl3': 12, 'N_4': 13, 'P': 14, 'F': 15, 'Cl': 16, 'Br': 17, 'I': 18,
                    'C_cat': 19, 'S_3': 20, 'S_o': 21, 'C_3X': 22, 'C_2X': 23, 'C_arX': 24, "X": 0}


def ligand_atom(name, nei_name):
    type_ = "X"
    if name == "C.3":
        for nei in nei_name:
            if nei in ['N', 'O', 'F', 'Cl', 'Br', 'I']:
                type_ = "C_3X"
                break
        if type_ == "X":
            type_ = "C_3"

    elif name == "C.2":
        for nei in nei_name:
            if nei in ['N', 'O', 'F', 'Cl', 'Br', 'I']:
                type_ = "C_2X"
                break
        if type_ == "X":
            type_ = "C_2"

    elif name == "C.1":
        type_ = "C_1"

    elif name == "C.ar":
        for nei in nei_name:
            if nei in ['N', 'O', 'F', 'Cl', 'Br', 'I']:
                type_ = "C_arX"
                break
        if type_ == "X":
            type_ = "C_ar"

    elif name == "O.3":
        for nei in nei_name:
            if nei in ['H']:
                type_ = "O_3"
                break
        if type_ == "X":
            type_ = "O_3p"

    elif name == "O.2":
        for nei in nei_name:
            if nei in ['P', 'S']:
                type_ = "O_2v"
                break
        if type_ == "X":
            type_ = "O_2"

    elif name == "O.co2":
        type_ = "O_co2"

    elif name[0] == "N":
        count = 0
        for nei in nei_name:
            if nei == "H":
                count += 1
        if count == 0:
            type_ = 'N_2'
        elif count == 1:
            type_ = 'N_am'
        elif count == 2:
            type_ = 'N_pl3'
        elif count == 3:
            type_ = 'N_pl3'
            # type_ = 'N_4'


    elif name == "P.3":
        type_ = "P"

    elif name == "F":
        type_ = "F"

    elif name == "Cl":
        type_ = "Cl"

    elif name == "Br":
        type_ = "Br"

    elif name == "I":
        type_ = "I"

    elif name == "C_cat":
        type_ = "C_cat"

    elif name == "S.3":
        type_ = "S_3"

    elif name == "S.2":
        type_ = "S_3"

    elif name == "S.o":
        type_ = "S_o"

    elif name == "S.o2":
        type_ = "S_o"

    return ligand_atom_type[type_]


protein_atom_type = {"C": 1, "Cs": 2, "CA": 3, "CB": 4, "CC": 5, "CN": 6, "CR": 7, "CT": 8, "CV": 9,
                     "CW": 10, "N": 11, "N2": 12, "N3": 13, "NA": 14, "NB": 15, "O": 16, "O2": 17, "OH": 18,
                     "S": 19, "SH": 20, "X": 0}


def protein_atom(name, res_name):
    type_ = "X"
    if name == "O" and res_name == "HOH":
        type_ = "OH"
    elif name == "C":
        type_ = "C"
    elif name == "O":
        type_ = "O"
    elif name == "N":
        type_ = "N"

    elif name == "NE" and res_name == "ARG":
        type_ = "N2"
    elif name == "CA" and res_name == "ARG":
        type_ = "CT"
    elif name == "CB" and res_name == "ARG":
        type_ = "CT"
    elif name == "CG" and res_name == "ARG":
        type_ = "CT"
    elif name == "CD" and res_name == "ARG":
        type_ = "CT"
    elif name == "CZ" and res_name == "ARG":
        type_ = "CA"
    elif name == "NH1" and res_name == "ARG":
        type_ = "N2"
    elif name == "NH2" and res_name == "ARG":
        type_ = "N2"

    elif name == "CA" and res_name == "ASN":
        type_ = "CT"
    elif name == "CB" and res_name == "ASN":
        type_ = "CT"
    elif name == "OD1" and res_name == "ASN":
        type_ = "O"
    elif name == "ND2" and res_name == "ASN":
        type_ = "N"
    elif name == "CG" and res_name == "ASN":
        type_ = "C"

    elif name == "CA" and res_name == "ASP":
        type_ = "CT"
    elif name == "CB" and res_name == "ASP":
        type_ = "CT"
    elif name == "OD1" and res_name == "ASP":
        type_ = "O2"
    elif name == "OD2" and res_name == "ASP":
        type_ = "O2"
    elif name == "CG" and res_name == "ASP":
        type_ = "C"

    elif name == "CA" and res_name == "CYS":
        type_ = "CT"
    elif name == "CB" and res_name == "CYS":
        type_ = "CT"
    elif name == "S" and res_name == "CYS":
        type_ = "SH"

    elif name == "CA" and res_name == "GLN":
        type_ = "CT"
    elif name == "CB" and res_name == "GLN":
        type_ = "CT"
    elif name == "OE1" and res_name == "GLN":
        type_ = "O"
    elif name == "NE2" and res_name == "GLN":
        type_ = "N"
    elif name == "CG" and res_name == "GLN":
        type_ = "CT"
    elif name == "CD" and res_name == "GLN":
        type_ = "C"

    elif name == "CA" and res_name == "GLU":
        type_ = "CT"
    elif name == "CB" and res_name == "GLU":
        type_ = "CT"
    elif name == "OE1" and res_name == "GLU":
        type_ = "O2"
    elif name == "OE2" and res_name == "GLU":
        type_ = "O2"
    elif name == "CG" and res_name == "GLU":
        type_ = "CT"
    elif name == "CD" and res_name == "GLU":
        type_ = "C"

    elif name == "CA" and res_name == "HIE":
        type_ = "CT"
    elif name == "CB" and res_name == "HIE":
        type_ = "CT"
    elif name == "ND1" and res_name == "HIE":
        type_ = "NB"
    elif name == "CD2" and res_name == "HIE":
        type_ = "CW"
    elif name == "CE1" and res_name == "HIE":
        type_ = "CR"
    elif name == "NE2" and res_name == "HIE":
        type_ = "NA"
    elif name == "CG" and res_name == "HIE":
        type_ = "CC"

    elif name == "CA" and res_name == "HID":
        type_ = "CT"
    elif name == "CB" and res_name == "HID":
        type_ = "CT"
    elif name == "ND1" and res_name == "HID":
        type_ = "NA"
    elif name == "CD2" and res_name == "HID":
        type_ = "CV"
    elif name == "CE1" and res_name == "HID":
        type_ = "CR"
    elif name == "NE2" and res_name == "HID":
        type_ = "NB"
    elif name == "CG" and res_name == "HID":
        type_ = "CC"

    elif name == "CA" and res_name == "HIS":
        type_ = "CT"
    elif name == "CB" and res_name == "HIS":
        type_ = "CT"
    elif name == "ND1" and res_name == "HIS":
        type_ = "NA"
    elif name == "CD2" and res_name == "HIS":
        type_ = "CV"
    elif name == "CE1" and res_name == "HIS":
        type_ = "CR"
    elif name == "NE2" and res_name == "HIS":
        type_ = "NB"
    elif name == "CG" and res_name == "HIS":
        type_ = "CC"

    elif name == "CA" and res_name == "HIP":
        type_ = "CT"
    elif name == "CB" and res_name == "HIP":
        type_ = "CT"
    elif name == "ND1" and res_name == "HIP":
        type_ = "NA"
    elif name == "CD2" and res_name == "HIP":
        type_ = "CV"
    elif name == "CE1" and res_name == "HIP":
        type_ = "CR"
    elif name == "NE2" and res_name == "HIP":
        type_ = "NA"
    elif name == "CG" and res_name == "HIP":
        type_ = "CC"

    elif name == "CB" and res_name == "ILE":
        type_ = "CT"
    elif name == "CA" and res_name == "ILE":
        type_ = "CT"
    elif name == "CG1" and res_name == "ILE":
        type_ = "CT"
    elif name == "CG2" and res_name == "ILE":
        type_ = "CT"
    elif name == "CD1" and res_name == "ILE":
        type_ = "CT"

    elif name == "CA" and res_name == "LEU":
        type_ = "CT"
    elif name == "CB" and res_name == "LEU":
        type_ = "CT"
    elif name == "CD1" and res_name == "LEU":
        type_ = "CT"
    elif name == "CD2" and res_name == "LEU":
        type_ = "CT"
    elif name == "CG" and res_name == "LEU":
        type_ = "CT"

    elif name == "CA" and res_name == "LYS":
        type_ = "CT"
    elif name == "CB" and res_name == "LYS":
        type_ = "CT"
    elif name == "NZ" and res_name == "LYS":
        type_ = "N3"
    elif name == "CG" and res_name == "LYS":
        type_ = "CT"
    elif name == "CD" and res_name == "LYS":
        type_ = "CT"
    elif name == "CE" and res_name == "LYS":
        type_ = "CT"

    elif name == "CA" and res_name == "MET":
        type_ = "CT"
    elif name == "CB" and res_name == "MET":
        type_ = "CT"
    elif name == "CE" and res_name == "MET":
        type_ = "CT"
    elif name == "S" and res_name == "MET":
        type_ = "S"
    elif name == "CG" and res_name == "MET":
        type_ = "CT"

    elif name == "CA" and res_name == "PHE":
        type_ = "CT"
    elif name == "CB" and res_name == "PHE":
        type_ = "CT"
    elif name == "CG" and res_name == "PHE":
        type_ = "CA"
    elif name == "CD1" and res_name == "PHE":
        type_ = "CA"
    elif name == "CD2" and res_name == "PHE":
        type_ = "CA"
    elif name == "CE1" and res_name == "PHE":
        type_ = "CA"
    elif name == "CE2" and res_name == "PHE":
        type_ = "CA"
    elif name == "CZ" and res_name == "PHE":
        type_ = "CA"

    elif name == "CA" and res_name == "PRO":
        type_ = "CT"
    elif name == "CB" and res_name == "PRO":
        type_ = "CT"
    elif name == "CD" and res_name == "PRO":
        type_ = "CT"
    elif name == "CG" and res_name == "PRO":
        type_ = "CT"

    elif name == "CA" and res_name == "SER":
        type_ = "CT"
    elif name == "CB" and res_name == "SER":
        type_ = "CT"
    elif name == "OG" and res_name == "SER":
        type_ = "OH"

    elif name == "CA" and res_name == "THR":
        type_ = "CT"
    elif name == "CB" and res_name == "THR":
        type_ = "CT"
    elif name == "OG1" and res_name == "THR":
        type_ = "OH"
    elif name == "CG2" and res_name == "THR":
        type_ = "CT"

    elif name == "CA" and res_name == "TRP":
        type_ = "CT"
    elif name == "CB" and res_name == "TRP":
        type_ = "CT"
    elif name == "CD1" and res_name == "TRP":
        type_ = "CW"
    elif name == "CD2" and res_name == "TRP":
        type_ = "CB"
    elif name == "NE1" and res_name == "TRP":
        type_ = "NA"
    elif name == "CE2" and res_name == "TRP":
        type_ = "CN"
    elif name == "CE2" and res_name == "TRP":
        type_ = "CN"
    elif name == "CE3" and res_name == "TRP":
        type_ = "CA"
    elif name == "CZ2" and res_name == "TRP":
        type_ = "CA"
    elif name == "CZ3" and res_name == "TRP":
        type_ = "CA"
    elif name == "CH2" and res_name == "TRP":
        type_ = "CA"
    elif name == "CG" and res_name == "TRP":
        type_ = "Cs"

    elif name == "CA" and res_name == "TYR":
        type_ = "CT"
    elif name == "CB" and res_name == "TYR":
        type_ = "CT"
    elif name == "CG" and res_name == "TYR":
        type_ = "CA"
    elif name == "CD1" and res_name == "TYR":
        type_ = "CA"
    elif name == "CD2" and res_name == "TYR":
        type_ = "CA"
    elif name == "CE1" and res_name == "TYR":
        type_ = "CA"
    elif name == "CE2" and res_name == "TYR":
        type_ = "CA"
    elif name == "CZ" and res_name == "TYR":
        type_ = "C"
    elif name == "OH" and res_name == "TYR":
        type_ = "OH"

    elif name == "CA" and res_name == "VAL":
        type_ = "CT"
    elif name == "CB" and res_name == "VAL":
        type_ = "CT"
    elif name == "CG1" and res_name == "VAL":
        type_ = "CT"
    elif name == "CG2" and res_name == "VAL":
        type_ = "CT"

    elif name == "CA" and res_name == "GLY":
        type_ = "CT"

    elif name == "CA" and res_name == "ALA":
        type_ = "CT"
    elif name == "CB" and res_name == "ALA":
        type_ = "CT"

    elif name == "CE":
        type_ = "CT"
    elif name == "OXT":
        type_ = "O"

    return protein_atom_type[type_]


# ========== 用于读取mol2文件原子信息 ==========  439不删除H，456删除H
def read_mol2_file(filename):
    atoms = []
    with open(filename, 'r') as f:
        all_line = f.read().split("\n")
        begin = False
        for line in all_line:
            if line.startswith("@<TRIPOS>ATOM"):
                begin = True
                continue
            if line.startswith("@<TRIPOS>BOND"):
                break
            if begin:
                atoms.append(line.rstrip())
    return atoms


def read_mol2_file_H(filename):
    atom_ = []
    atomlist = read_mol2_file(filename)
    for atom in atomlist:
        atomitem = atom.split()

        if len(atomitem) != 10:
            atomitem.append("DICT")

        atom_.append(atomitem)
    return np.array(atom_)


def read_mol2_file_withoutH(filename):
    atom_ = []
    atomlist = read_mol2_file(filename)
    for atom in atomlist:
        atomitem = atom.split()

        if len(atomitem) != 10:
            atomitem.append("DICT")

        if atomitem[1][0] != "H":
            atom_.append(atomitem)
    return np.array(atom_)


# ========= 提取相互作用力csv文件信息 ==========
def InterActionDict(csv_file_name="/home/yujie/8ICJ-TTP_ligand0_pv_interactions.csv"):
    # input: 一个记录相互作用力指纹的csv文件
    # output:
    # defaultdict(list,
    #             {'Hbond': [[27, 'ARG179', 'HE'],
    #                        [27, 'ARG179', 'HH21'],
    #                        [59, 'CYS285', 'SG'],
    #                        [59, 'SER339', 'O'],
    #                        [11, 'TRP348', 'HE1']],
    #              'Salt': [[27, 'ARG179', 'NH2']],
    #              'PiPi': [[4, 'TRP340', 'CD2']],
    #              'HPhob': [[32, 'MET176', 'SD'],
    #                        [40, 'MET176', 'SD'],
    #                        [40, 'MET176', 'CE'],
    #                        [40, 'GLU239', 'CG'],
    #                        [40, 'PHE244', 'CE1'],
    #                        [40, 'PHE244', 'CZ'],
    #                        [30, 'CYS285', 'CB'],
    #                        [31, 'CYS285', 'CB'],
    #                        [36, 'THR288', 'CG2'],
    #                        [43, 'TYR338', 'CB'],
    #                        [41, 'TYR338', 'CG'],
    #                        [43, 'TYR338', 'CG'],
    #                        [41, 'TYR338', 'CD1'],
    #                        [31, 'TYR338', 'CE1'],
    #                        [42, 'TYR338', 'CE1'],
    #                        [43, 'TRP340', 'CG'],
    #                        [43, 'TRP340', 'CD2'],
    #                        [13, 'TRP340', 'CE3'],
    #                        [43, 'TRP340', 'CE3'],
    #                        [43, 'TRP340', 'CZ2'],
    #                        [5, 'TRP340', 'CZ3'],
    #                        [6, 'TRP340', 'CZ3'],
    #                        [14, 'TRP340', 'CZ3'],
    #                        [43, 'TRP340', 'CZ3'],
    #                        [43, 'TRP340', 'CH2'],
    #                        [42, 'PHE381H', 'CD1'],
    #                        [1, 'PHE381H', 'CD2'],
    #                        [42, 'PHE381H', 'CE1']]})

    # function
    # 将提取的相互作用力指纹csv文件，制备成一个字典，值为列表，列表中每一个元素为一个相互作用力，
    # 也是一个列表，索引0为小分子原子薛定谔序号（1-based）

    df = pd.read_csv(csv_file_name)
    interaction = defaultdict(list)

    type_ = ['Hbond' if i[0:3] in ['HAc', 'HDo'] else 'PiPi' if i in ['PiEdge', 'PiFace'] else i for i in
             df.Type.values]  # ['Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Hbond', 'Salt', 'Salt', 'Salt', 'Salt', 'PiFace', 'HPhob', 'HPhob', 'HPhob', 'HPhob', 'HPhob', 'HPhob']

    ligand_atom = [int(i.split("(")[0]) for i in
                   df.RecAtom.values]  # ['2', '6', '7', '11', '11', '12', '12', '18', '26', '38', '10', '10', '11', '11', '21', '19', '19', '19', '27', '28', '28']

    residue_num = [i.split("(")[0].split(':')[1].strip(" ") for i in
                   df.LigResidue.values]  # ['560', '564', '415', '414', '482', '482', '560', '416', '4', '4', '1002', '1009', '482', '486', '115', '416', '416', '416', '115', '115', '115']

    residue_name = [i.split("(")[-1][:-1].strip(" ") for i in
                    df.LigResidue.values]  # ['LYS', 'ASN', 'LYS', 'SER', 'ARG', 'ARG', 'LYS', 'TYR', 'DA', 'DA', 'CA', 'CA', 'ARG', 'LYS', 'DT', 'TYR', 'TYR', 'TYR', 'DT', 'DT', 'DT']

    atom_type = [i.split("(")[-1][:-1].strip(" ") for i in df.LigAtom.values]

    for bondtype, ligatom, resnum, resname, atomtype in zip(type_, ligand_atom, residue_num, residue_name, atom_type):
        interaction[bondtype].append([ligatom, resname+resnum, atomtype])

    return interaction


# ========= 重设ligand原子索引 ==========
def ResetMolIndex(ligand, index, mol_file_H, mol_file):
    # input:
    # Chem.Mol; 原子的1-based索引

    # output:
    # 原子 0-based的索引；当原子类型为H的时候，返回该原子所连重原子的0-based索引  并且所有的sdf文件小分子索引都是在最后的，所以有H和没有H，重原子的索引是一样的

    # function
    # 1、将薛定谔的编码，变成rdkit的编码
    # 2、如果目标原子是一个H，把编码转化为其所连的重原子的编码

    #     ligand = rdkit.Chem.MolFromMolFile(ligand_sdf_file)
    #     ligand = Chem.AddHs(ligand)
    atom_token = mol_file_H[index-1]
    if atom_token[1][0] != "H":
        idx = np.where((mol_file == atom_token).all(axis=1))[0]
    else:
        atom = ligand.GetAtomWithIdx(index - 1)
        if len(atom.GetNeighbors()) != 1 or atom.GetNeighbors()[0].GetSymbol() == 'H':
            return None
        else:
            token = mol_file_H[atom.GetNeighbors()[0].GetIdx()]  # 就是H的邻居原子的所在行信息
            idx = np.where((mol_file == token).all(axis=1))[0]

    return idx


# ========= 重设pocket原子索引 ==========
def ResetPocketAtom(pock, res_type, atom_type, mol_file_H, mol_file):
    # input：
    # 蛋白口袋Chem.Mol; 残基识别符：‘PHE381H’; 原子识别符号：’CD1‘; mol2文件原子信息含H, mol2文件原子信息不含H

    # output: 直接返回没有H原子的时候蛋白原子的索引
    #

    # function
    # 1、对于原本就为重原子的原子而言，直接返回删去原子序号的 atom_identity_string
    # 2、对于H的返回其相邻原子的atom_identity_string
    if atom_type[0] != "H":
        idx = np.intersect1d(np.where(mol_file[:, 1] == atom_type), np.where(mol_file[:, 7] == res_type))
    else:
        idx = np.intersect1d(np.where(mol_file_H[:, 1] == atom_type), np.where(mol_file_H[:, 7] == res_type))
        
        if len(idx) != 1:
            return None

        atom = pock.GetAtomWithIdx(int(idx))
        if len(atom.GetNeighbors()) != 1 or atom.GetNeighbors()[0].GetSymbol() == 'H':
            return None
        else:
            token = mol_file_H[atom.GetNeighbors()[0].GetIdx()]  # 就是H的邻居原子的所在行信息
            idx = np.where((mol_file == token).all(axis=1))[0]

    return idx


# ========== 制作相互作用力编码 ==========   但是似乎没有什么作用了
def InteractionEmbedding(interactions_file):

    interactions = InterActionDict(interactions_file)

    dic = {'HPhob': 0, 'Hbond': 0, 'PiCat': 0, 'PiPi': 0, 'Salt': 0, 'XBond': 0}
    embedding = []
    for key in interactions.keys():
        num_key = len(interactions[key])

        dic[key] = num_key

    for key_ in dic.keys():
        embedding.append(dic[key_])

    return embedding


def GetAtomPairAndType(ligand_file="/home/yujie/leadopt/data/ic50_final_pose/5AUU-LU2/5AUU-LU2_ligand4.sdf", rmH=True):
    # input：
    # 相互作用力字典，三个文件

    # output：
    # [ligand atom，pocket atom]，[interatcion type]

    interactions_file = ligand_file.rsplit('.', 1)[0] + "_pv_interactions.csv"
    pocket_file = ligand_file.rsplit('/', 1)[0] + '/pocket.mol2'
    pocket_pdbfile = ligand_file.rsplit('/', 1)[0] + '/pocket.pdb'
    ligand_mol2file = ligand_file.rsplit('.', 1)[0] + ".mol2"

    # 读取相互作用力
    interactions = InterActionDict(interactions_file)

    # 读取分子，与蛋白
    ligand = rdkit.Chem.MolFromMolFile(ligand_file, removeHs=False)   # 需要查看 不会有有None，以及和文本读取的长度是否相同，下面的口袋也一样
    if ligand is None:
        ligand = rdkit.Chem.MolFromMol2File(ligand_mol2file, removeHs=False)

    pock = rdkit.Chem.MolFromPDBFile(pocket_pdbfile, sanitize=True, removeHs=False)


    # 文本文件读取
    lig_file_H = read_mol2_file_H(ligand_mol2file)
    lig_file = read_mol2_file_withoutH(ligand_mol2file)

    pock_file_H = read_mol2_file_H(pocket_file)
    pock_file = read_mol2_file_withoutH(pocket_file)

    # 读取原子ids
    atom_pair = []
    interactions_type = []
    for key in interactions.keys():
        for item_ligand in interactions[key]:  # item_ligand  = [27, 'ARG179', 'HE']  其中27是薛定谔的 是1-based的

            idx_pock = ResetPocketAtom(pock, item_ligand[1], item_ligand[2], pock_file_H, pock_file)
            idx_lig = ResetMolIndex(ligand, item_ligand[0], lig_file_H, lig_file)

            if idx_lig is None or idx_pock is None:
                continue

            if len(idx_lig) != 1 or len(idx_pock) != 1:
                continue

            atom_pair.append((int(idx_lig), int(idx_pock)))
            interactions_type.append(key)

    return atom_pair, interactions_type


# =================== 化合物信息表征 =====================
INTERACTION_TYPES = [
    "saltbridge",
    "hbonds",
    "pication",
    "pistack",
    "halogen",
    "waterbridge",
    "hydrophobic",
    "metal_complexes",
]

# ============ 主族与周期 ===========
pt = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""
PERIODIC_TABLE = dict()
for i, per in enumerate(pt.split()):
    for j, ele in enumerate(per.split(",")):
        PERIODIC_TABLE[ele] = (i, j)   # 第i主族，第j周期
PERIODS = [0, 1, 2, 3, 4, 5]
GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    x = list(map(lambda s: x == s, allowable_set))
    x = [int(i) for i in x]
    return x


def get_period_group(atom) -> List[bool]:
    period, group = PERIODIC_TABLE[atom.GetSymbol().upper()]
    return one_of_k_encoding(period, PERIODS) + one_of_k_encoding(group, GROUPS)   # 共6+18维度

# =========== 常规特征 ============
class Featurization_parameters:
    """
    A class holding molecule featurization parameters as attributes.
    """

    def __init__(self) -> None:
        # Atom feature sizes
        # self.SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br"]
        self.MAX_ATOMIC_NUM = 100
        self.ATOM_FEATURES = {
            'atomic_num': ["C", "N", "O", "S", "F", "P", "Cl", "Br"],
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'chiral_tag': [0, 1, 2, 3],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
        }

        # Distance feature sizes
        # self.PATH_DISTANCE_BINS = list(range(10))
        # self.THREE_D_DISTANCE_MAX = 20
        # self.THREE_D_DISTANCE_STEP = 1
        # self.THREE_D_DISTANCE_BINS = list(range(0, self.THREE_D_DISTANCE_MAX + 1, self.THREE_D_DISTANCE_STEP))

        # len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
        self.ATOM_FDIM = sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        self.EXTRA_ATOM_FDIM = 0
        self.BOND_FDIM = 14
        self.EXTRA_BOND_FDIM = 0
        self.EXPLICIT_H = False
        self.ADDING_H = False


HBOND_DONOR_INDICES = ["[!#6;!H0]"]
HBOND_ACCEPPTOR_SMARTS = [
    "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]

# ============ 是否为H键供体和受体 ===========
def get_hbond_atom_indices(mol, smarts_list: List[str]) -> np.ndarray:
    indice = []
    for smarts in smarts_list:
        smarts = Chem.MolFromSmarts(smarts)
        indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
    indice = np.array(indice)
    return indice

# =========== 范德华半径 ===========
def get_vdw_radius(atom):
    atomic_num = atom.GetAtomicNum()
    return Chem.GetPeriodicTable().GetRvdw(atomic_num)


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


PARAMS = Featurization_parameters()


def atom_features(atom: Chem.rdchem.Atom, H_bond_information: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    if atom is None:
        features = [0] * PARAMS.ATOM_FDIM
    else:
        features = onek_encoding_unk(atom.GetSymbol(), PARAMS.ATOM_FEATURES['atomic_num']) + \
                   onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
                   onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
                   onek_encoding_unk(int(atom.GetChiralTag()), PARAMS.ATOM_FEATURES['chiral_tag']) + \
                   onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
                   onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
                   get_period_group(atom) + \
                   [atom.GetMass() * 0.01] + \
                   [atom.GetExplicitValence() * 0.1] + \
                   [atom.GetImplicitValence() * 0.1] + \
                   [get_vdw_radius(atom)] + \
                   [1 if atom.GetIsAromatic() else 0]
        # scaled to about the same range as other features
        if H_bond_information is not None:
            features += H_bond_information
    return features


# 9+7+6+5+6+6+24 + 7

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: An RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [0] + [0] * (PARAMS.BOND_FDIM - 1)  # 非共价键
    else:
        bt = bond.GetBondType()
        fbond = [
            1,  # 共价键
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


# ============= 识别芳香中心 ==============
def get_aromatic_rings(mol) -> list:
    ''' return aromaticatoms rings'''
    aromaticity_atom_id_set = set()
    rings = []
    for atom in mol.GetAromaticAtoms():
        aromaticity_atom_id_set.add(atom.GetIdx())
    # get ring info
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        ring_id_set = set(ring)
        # check atom in this ring is aromaticity
        if ring_id_set <= aromaticity_atom_id_set:
            rings.append(list(ring))
    return rings

#    这里mol 是不含H的哦
def bond_matrix(mol, lig_or_pock="lig", mol2_file=None,mol_H=None):        
    
    # 配体的mol2_file需要带有H
    
    atom1 = []
    atom2 = []
    a_features = []
    b_features = []
    a_coordinates = []
    atom_type = []

    rings = get_aromatic_rings(mol)  # 芳香环的识别
    donor_index = get_hbond_atom_indices(mol, HBOND_DONOR_INDICES)
    acceptor_index = get_hbond_atom_indices(mol, HBOND_ACCEPPTOR_SMARTS)

    num_atoms = len(mol.GetAtoms())
    num_aromatic = len(rings)


    if lig_or_pock == "lig":
        num_atoms_H = len(mol_H.GetAtoms())

        for a1 in range(num_atoms_H):
            nei_name = []
            name = mol2_file[a1][5]
            if name != "H":
                for nei in mol_H.GetAtoms()[a1].GetNeighbors():
                    nei_name.append(nei.GetSymbol())
                idx = ligand_atom(name, nei_name)                           
                atom_type.append(idx)


    for a1 in range(num_atoms):

        H_bond_information = [1 if a1 in donor_index else 0] + [1 if a1 in acceptor_index else 0]

        af = atom_features(mol.GetAtoms()[a1], H_bond_information)
        a_features.append(af)

        x, y, z = mol.GetConformer().GetAtomPosition(a1)
        a_coordinates.append([x, y, z])

        # if lig_or_pock == "lig":
        #     nei_name = []
        #     name = mol2_file[a1][5]
        #     for nei in mol.GetAtoms()[a1].GetNeighbors():
        #         nei_name.append(nei.GetSymbol())
        #     idx = ligand_atom(name, nei_name)                               # 准确说是Id 不是index
        #     atom_type.append(idx)

        if lig_or_pock == "pock":
            name = mol2_file[a1][1]
            res_name = mol2_file[a1][7][0:3]
            idx = protein_atom(name, res_name)                             # 准确说是Id 不是index
            atom_type.append(idx)

        for a2 in range(a1 + 1, num_atoms):
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                continue

            # adj[a1,a2] = 1
            # adj[a2,a1] = 1
            bf = bond_features(bond)
            b_features.append(bf)
            b_features.append(bf)

            atom1.append(a1)
            atom2.append(a2)
            atom1.append(a2)
            atom2.append(a1)

    a_features = np.array(a_features)
    b_features = np.array(b_features)
    a_coordinates = np.array(a_coordinates)

    # ============= 加入芳香中心，每一个芳香环多一个原子(中心)，坐标为芳香环均值，中心和芳香环中各个原子建边，属性全为0 ==========
    for i, ring in enumerate(rings):
        # adj的改变
        # adj[num_atoms+i,:][ring] = 1
        # adj[:,num_atoms+i][ring] = 1
        # adj[num_atoms+i,:][num_atoms+i] = 1
        # 链接关系的改变

        # 芳香中心当作芳香6元环的C来处理
        if lig_or_pock == "lig":
            atom_type.append(4)
        elif lig_or_pock == "pock":
            atom_type.append(3)

        # 加边
        for a_ in ring:
            atom1.append(num_atoms + i)
            atom2.append(a_)
            atom1.append(a_)
            atom2.append(num_atoms + i)
        # 加入坐标
        a_coordinates = np.concatenate([a_coordinates, np.mean(a_coordinates[ring], axis=0, keepdims=True)], axis=0)
        # 边的属性和atom属性
        a_features = np.concatenate([a_features, np.array([0] * (a_features.shape[1]))[np.newaxis]], axis=0)
        b_features = np.concatenate([b_features, np.zeros([2 * len(ring), b_features.shape[1]])], axis=0)  # 当作共价键来处理

    a_features_hbond = a_features[:, 66:70]

    return atom1, atom2, a_features, b_features, a_coordinates, a_features_hbond, atom_type


def Graph_Information(ligand_file, Interaction_Bond_Build=False, rmH=True,knowledge=None):
    # file directory determination
    interactions_file = ligand_file.rsplit('.', 1)[0] + "_pv_interactions.csv"
    pocket_file = ligand_file.rsplit('/', 1)[0] + '/pocket.mol2'
    pocket_pdbfile = ligand_file.rsplit('/', 1)[0] + '/pocket.pdb'
    ligand_file2 = ligand_file.rsplit(".", 1)[0] + ".mol2"

    # ligand and pocket file read in
    # ligand = rdkit.Chem.MolFromMolFile(ligand_file,removeHs = True)
    ligand = rdkit.Chem.MolFromMolFile(ligand_file, removeHs=True)  # 需要查看 不会有有None，以及和文本读取的长度是否相同，下面的口袋也一样
    if ligand is None:
        ligand = rdkit.Chem.MolFromMol2File(ligand_file2, removeHs=True)

    pock = rdkit.Chem.MolFromPDBFile(pocket_pdbfile, sanitize=True, removeHs=True)


    num_atoms = len(ligand.GetAtoms())
    mol2_file_pock = read_mol2_file_withoutH(pocket_file)
    mol2_file_lig = read_mol2_file_H(ligand_file2)

    ligand_H = rdkit.Chem.MolFromMolFile(ligand_file, removeHs=False)  # 需要查看 不会有有None，以及和文本读取的长度是否相同，下面的口袋也一样
    if ligand_H is None:
        ligand_H = rdkit.Chem.MolFromMol2File(ligand_file2, removeHs=False)



    # merge ligand and pocket
    # complex_ = rdkit.Chem.rdmolops.CombineMols(lig,pock)
    # num_atoms_ = len(complex_.GetAtoms())

    # the element to induce a dgl Graph (without interaction bond)
    atom1_lig, atom2_lig, a_features_lig, b_features_lig, a_coordinates_lig, a_features_hbond_lig, atom_type_lig = bond_matrix(
        ligand, lig_or_pock="lig", mol2_file=mol2_file_lig, mol_H = ligand_H)
    atom1_pock, atom2_pock, a_features_pock, b_features_pock, a_coordinates_pock, a_features_hbond_pock, atom_type_pock = bond_matrix(
        pock, lig_or_pock="pock", mol2_file=mol2_file_pock)

    # interaction embedding (6-dim)
    # embedding = InteractionEmbedding(interactions_file)
    # embeddings = [embedding for i in range(len(a_features_lig))]
    # embeddings2 = [embedding for i in range(len(a_features_pock))]

    atom1_lig, atom2_lig = torch.tensor(atom1_lig), torch.tensor(atom2_lig)
    g_lig = dgl.graph((atom1_lig, atom2_lig), num_nodes=len(a_features_lig))

    g_lig.ndata['atom_coordinate'] = torch.tensor(a_coordinates_lig, dtype=torch.float32)
    g_lig.ndata['atom_feature'] = torch.tensor(a_features_lig, dtype=torch.float32)
    g_lig.edata['edge_feature'] = torch.tensor(b_features_lig, dtype=torch.float32)
    g_lig.ndata['atom_feature_hbond'] = torch.tensor(a_features_hbond_lig, dtype=torch.float32)
    # g_lig.ndata['interaction_embedding'] = torch.tensor(embeddings, dtype=torch.float32)
    g_lig.ndata["p_or_l"] = torch.tensor([[0] for _ in range(len(a_features_lig))], dtype=torch.float32)
    g_lig.ndata["index"] = torch.tensor(atom_type_lig, dtype=torch.float32)
    g_lig.edata['attention_weight'] = torch.tensor([[1.] for _ in range(len(b_features_lig))], dtype=torch.float32)
    g_lig.edata['schrodinger'] = torch.tensor([[0] for _ in range(len(b_features_lig))], dtype=torch.float32)

    atom1_pock, atom2_pock = torch.tensor(atom1_pock), torch.tensor(atom2_pock)
    g_pock = dgl.graph((atom1_pock, atom2_pock), num_nodes=len(a_features_pock))

    g_pock.ndata['atom_coordinate'] = torch.tensor(a_coordinates_pock, dtype=torch.float32)
    g_pock.ndata['atom_feature'] = torch.tensor(a_features_pock, dtype=torch.float32)
    g_pock.edata['edge_feature'] = torch.tensor(b_features_pock, dtype=torch.float32)
    g_pock.ndata['atom_feature_hbond'] = torch.tensor(a_features_hbond_pock, dtype=torch.float32)
    g_pock.ndata["p_or_l"] = torch.tensor([[1] for _ in range(len(a_features_pock))], dtype=torch.float32)
    # g_pock.ndata['interaction_embedding'] = torch.tensor(embeddings2, dtype=torch.float32)
    g_pock.ndata["index"] = torch.tensor(atom_type_pock, dtype=torch.float32)
    g_pock.edata['attention_weight'] = torch.tensor([[1.] for _ in range(len(b_features_pock))], dtype=torch.float32)
    g_pock.edata['schrodinger'] = torch.tensor([[0] for _ in range(len(b_features_pock))], dtype=torch.float32)
    g_complex = dgl.batch([g_lig, g_pock])

    # 距离边
    # dist_nei = distance_matrix(g_lig.ndata['atom_coordinate'],g_lig.ndata['atom_coordinate'])
    # node_idx = np.where(dist_nei <= 2.5)

    # for step in range(len(node_idx[0])):
    #     i = list(zip(node_idx[0],node_idx[1]))[step]
    #     # 配体本身有的共价边，不要重复添加
    #     if i not in zip(atom1_lig,atom2_lig):
    #         g_complex.add_edges(node_idx[0][step], node_idx[1][step])
    #             g_complex.add_edges(node_idx[1][step], node_idx[0][step])

    dist_wai = distance_matrix(g_pock.ndata['atom_coordinate'], g_lig.ndata['atom_coordinate'])
    node_idx = np.where(dist_wai <= 5)

    for step in range(len(node_idx[0])):
        pock_num = g_pock.ndata['index'][node_idx[0][step]]
        ligand_num = g_lig.ndata['index'][node_idx[1][step]]

        if pock_num != 0 and ligand_num != 0:
            final_num = (pock_num - 1) * 24 + ligand_num
            dist_ = dist_wai[node_idx[0][step]][node_idx[1][step]]
            D = knowledge["Pairwise_distance (Å)"].values
            first = knowledge[f'Unnamed: {int(final_num)}'].values
            weight_ = np.interp(dist_, D, first)

        else:
            final_num = 20
            dist_ = dist_wai[node_idx[0][step]][node_idx[1][step]]
            D = knowledge["Pairwise_distance (Å)"].values
            first = knowledge[f'Unnamed: {int(final_num)}'].values
            # weight_ = np.interp(dist_, D, first)
            weight_ = 0.8

        weight_ = torch.tensor(weight_, dtype=torch.float32)
        g_complex.add_edges(node_idx[0][step] + len(a_coordinates_lig), node_idx[1][step])
        g_complex.add_edges(node_idx[1][step], node_idx[0][step] + len(a_coordinates_lig))

        g_complex.edata["attention_weight"][
            g_complex.edge_ids(node_idx[0][step] + len(a_coordinates_lig), node_idx[1][step],
                               return_uv=False)] = weight_
        g_complex.edata["attention_weight"][
            g_complex.edge_ids(node_idx[1][step], node_idx[0][step] + len(a_coordinates_lig),
                               return_uv=False)] = weight_

    # ============薛定谔计算所得非共价作用力边===============
    if Interaction_Bond_Build == True:

        atom_pair, interactions_type = GetAtomPairAndType(ligand_file)

        # bond = None
        # None_bond_feature = bond_features(bond) # 只有第一维度是1 表示共价键
        # pp = torch.tensor(None_bond_feature,dtype=torch.float64)

        for pair in atom_pair:
            # g_complex.add_edges(pair[0],pair[1]+len(a_coordinates_lig))
            # g_complex.add_edges(pair[1]+len(a_coordinates_lig),pair[0])

            # 需要判断是否存在边
            if g_complex.has_edges_between(pair[0], pair[1] + len(a_coordinates_lig)):
                id1 = g_complex.edge_ids(pair[0], pair[1] + len(a_coordinates_lig), return_uv=False)
                id2 = g_complex.edge_ids(pair[1] + len(a_coordinates_lig), pair[0], return_uv=False)
                g_complex.edata['schrodinger'][id1] = 1.0
                g_complex.edata['schrodinger'][id2] = 1.0

            # atom1.append(pair[0])
            # atom1.append(pair[1])
            # atom2.append(pair[0])
            # atom2.append(pair[1])
            # b_features.append(None_bond_feature)
            # b_features.append(None_bond_feature)

    g_complex = dgl.remove_self_loop(g_complex)

    return g_complex, g_pock


def graph_save(ligand_file, pickle_file, pickle_file2, Interaction_Bond_Build=False, rmH=True,knowledge=None):

    # if not os.path.exists(pickle_file):
    g_complex, g_pock = Graph_Information(ligand_file, Interaction_Bond_Build, rmH, knowledge)
    pickle_file = open(pickle_file, 'wb')
    pickle.dump(g_complex, pickle_file)
    pickle_file.close()

    if not os.path.exists(pickle_file2):
        pickle_file2 = open(pickle_file2, 'wb')
        pickle.dump(g_pock, pickle_file2)
        pickle_file2.close()


# def parallel(z):
#     return save_rmsd_to_csv(z[0], z[1], z[2])


def print_error(value):
    print("error: ", value)


if __name__ == "__main__":

# 2022 11 02 训练集的处理
    setup_cpu(10)

    PARAMS = Featurization_parameters()

    Interaction_Bond_Build = False
    rmH = True
    knowledge = pd.read_excel("/home/yujie/leadopt/data/FEP221102/GARF_probability_distribution.xlsx")

    data_list = []

    files = [i for i in os.listdir('/home/yujie/leadopt/data/selection/') if i.split("_")[-1] == 'pose'] 

    # os.system(f'mkdir "/home/yujie/leadopt/data/result/test_set_fep+_graph_rmH_IV/"')

    # for i in files:
    #     os.system(f'mkdir {"/home/yujie/leadopt/data/ic50_graph_rmH_221102/" + i}')


    for i in files:

        i_ = i.split('_')[0]
        i_pose = i

        file = [i_ + '/' + a for a in os.listdir("/home/yujie/leadopt/data/selection/" + i_pose) if a.rsplit(".", 1)[-1] == 'sdf']

        file2 = [i_pose +'/'+ a for a in os.listdir("/home/yujie/leadopt/data/selection/" + i_pose) if a.rsplit(".", 1)[-1] == 'sdf']


        ligand_file = ["/home/yujie/leadopt/data/selection/" + c  for c in file2]
        pickle_file = ['/home/yujie/leadopt/data/selection_graph/' + c.rsplit('.', 1)[0] + '.pkl' for c in file]
        pickle_file2 = ["/home/yujie/leadopt/data/selection_graph/" + c.rsplit('/', 1)[0] + '/pocket.pkl' for c in file]

        for g in range(len(ligand_file)):
            data_list.append([ligand_file[g], pickle_file[g], pickle_file2[g], Interaction_Bond_Build, rmH, knowledge])


    pool = multiprocessing.Pool(7)

    for i in data_list:
        pool.apply_async(graph_save, i, error_callback=print_error)

        # res = pool.map(parallel, mcss_size)

    pool.close()
    pool.join()









# 2022 11 02 FEP1和FEP2的处理
    # setup_cpu(10)

    # PARAMS = Featurization_parameters()

    # Interaction_Bond_Build = False
    # rmH = True
    # knowledge = pd.read_excel("/home/yujie/leadopt/data/FEP221102/GARF_probability_distribution.xlsx")

    # data_list = []

    # # "/home/yujie/leadopt/data/test_set_fep/"
    # # files = ['enpp1_pose']

    # # os.system(f'mkdir "/home/yujie/leadopt/data/result/systems_graph/"')

    # # for i in files:
    # #     os.system(f'mkdir {"/home/yujie/leadopt/data/result/systems_graph/" + i}')

    # FEP1 = ['Tyk2', 'p38', 'MCL1', 'Jnk1', 'Thrombin', 'CDK2', 'Bace', 'PTP1B']
    # FEP2 = ['shp2', 'syk', 'cdk8', 'eg5', 'pfkfb3', 'cmet', 'hif2a', 'tnks2']

    # file_name = FEP2

    # for file_ in file_name:

    #     file = [a for a in os.listdir(f'/home/yujie/leadopt/data/FEP221102/FEP2_221102/{file_}') if
    #             a.rsplit(".", 1)[-1] == 'sdf']

    #     ligand_file = [f"/home/yujie/leadopt/data/FEP221102/FEP2_221102/{file_}/" + c for c in file]
    #     pickle_file = [f"/home/yujie/leadopt/data/FEP221102/FEP2_G_221102/{file_}/" + c.rsplit('.', 1)[0] + '.pkl' for c in file]
    #     pickle_file2 = [f"/home/yujie/leadopt/data/FEP221102/FEP2_G_221102/{file_}/" + 'pocket.pkl' for c in
    #                     file]

    #     for g in range(len(ligand_file)):
    #         data_list.append([ligand_file[g], pickle_file[g], pickle_file2[g], Interaction_Bond_Build, rmH, knowledge])

    # pool = multiprocessing.Pool(7)

    # for i in data_list:
    #     pool.apply_async(graph_save, i, error_callback=print_error)

    #     # res = pool.map(parallel, mcss_size)

    # pool.close()
    # pool.join()
