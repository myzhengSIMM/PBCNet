import rdkit
from rdkit import Chem
import numpy as np
import pandas as pd
from collections import defaultdict


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


def get_aromatic_rings(mol) -> list:
    ''' return aromaticatoms rings'''
    aromaticity_atom_id_set = set()
    rings = []
    for atom in mol.GetAromaticAtoms():
        aromaticity_atom_id_set.add(atom.GetIdx())
    # get ring info
    ssr = rdkit.Chem.GetSymmSSSR(mol)
    for ring in ssr:
        ring_id_set = set(ring)
        # check atom in this ring is aromaticity
        if ring_id_set <= aromaticity_atom_id_set:
            rings.append(list(ring))
    return rings


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
    ligand = Chem.MolFromMolFile(ligand_file, removeHs=False)   # 需要查看 不会有有None，以及和文本读取的长度是否相同，下面的口袋也一样
    if ligand is None:
        ligand = Chem.MolFromMol2File(ligand_mol2file, removeHs=False)

    pock = Chem.MolFromPDBFile(pocket_pdbfile, sanitize=True, removeHs=False)


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

            # atom_pair.append([int(idx_lig), int(idx_pock)])
            atom_pair.append([int(idx_lig), item_ligand[1]+item_ligand[2]])

            interactions_type.append(key)

    mol = rdkit.Chem.MolFromMolFile(ligand_file, removeHs=True)
    rings = get_aromatic_rings(mol)  # 芳香环的识别

    a_coordinates = []
    num_atoms = len(mol.GetAtoms())

    num_aromatic = len(rings)

    for a1 in range(num_atoms):
        x, y, z = mol.GetConformer().GetAtomPosition(a1)
        a_coordinates.append([x, y, z])

    # for i in atom_pair:
        # i[1] = i[1] + len(a_coordinates) + num_aromatic
        #+ num_aromatic

    return atom_pair, interactions_type

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