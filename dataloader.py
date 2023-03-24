import json
import pandas as pd

import os
from collections import OrderedDict

import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from Bio.PDB import *
import deepchem
import pickle


pk = deepchem.dock.ConvexHullPocketFinder()


max_d = 150 #预定义药物SMILES最大长度
max_p = 1200 #预定义蛋白质氨基酸最大长度



Seq_Set = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

Seq_Set_len = 25

# CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
#                  ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
#                  "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
#                  "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
#                  "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
#                  "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
#                  "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
#                  "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
#                  "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
#                  "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
#                  "t": 61, "y": 62}
# CHARCANSMILEN = 62



SMILES_Set = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
SMILES_Set_len= 64

def get_pdb(task): #获取蛋白质的pdb文件，以处理蛋白质的binding pockets
    with open(task + '/protein_pdb.pkl', 'rb') as f:
        protein_pdb_dict = pickle.load(f)

    protein_pdb_dict = dict(protein_pdb_dict)
    protein_pdb = []

    for value in protein_pdb_dict.values():
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(
            value, pdir='./pdbs/', overwrite=True, file_format="pdb"
        )
        # Rename file to .pdb from .ent
        os.rename(
            './pdbs/' + "pdb" + value + ".ent", './pdbs/' + value + ".pdb"
        )
        protein_pdb.append(value)

    return protein_pdb

#处理蛋白特征
def process_protein(pdb_file): 
    m = Chem.MolFromPDBFile(pdb_file)
    am = GetAdjacencyMatrix(m)
    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)

        ami = am[np.array(idxs)[:, None], np.array(idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_matrix(ami)
        graph = dgl.DGLGraph(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)

    constructed_graphs = dgl.batch(constructed_graphs)

    # amj = am[np.array(not_in_binding)[:, None], np.array(not_in_binding)]
    # not_binding_atoms = []
    # for item in not_in_binding:
    #     not_binding_atoms.append((m.GetAtoms()[item], d2[item]))
    # H = get_atom_feature(not_binding_atoms)
    # g = nx.convert_matrix.from_numpy_matrix(amj)
    # graph = dgl.DGLGraph(g)
    # graph.ndata['h'] = torch.Tensor(H)
    # graph = dgl.add_self_loop(graph)
    # constructed_graphs = dgl.batch([constructed_graphs, graph])
    return binding_parts, not_in_binding, constructed_graphs

#处理序列
def one_of_k_encoding(x, allowable_set): 
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

#进行蛋白质原子特征处理
def atom_feature(atom): 
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', "other"]) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])


#获取原子特征
def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature(m[i][0]))
    H = np.array(H)

    return H



def orderdict_list(dict):
    x = []
    for d in dict.keys():
        x.append(dict[d])
    return x

#one-hot处理SMILES序列
def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X

#one-hot处理氨基酸序列
def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X

#矩阵式处理SMILES序列
def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X

#矩阵式处理氨基酸序列
def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X

#获取超出预定义序列长度的药物和蛋白质的列表
def get_removelist(list_name, length):
    removelist = []
    # Davis  SMILES:85 protein:1200   KIBA   SMILES:100   protein:1200
    for i, x in enumerate(list_name):
        if len(x) >= length:
            removelist.append(i)
    return removelist

#删除列表
def list_remove(list_name, removelist):
    a_index = [i for i in range(len(list_name))]
    a_index = set(a_index)
    b_index = set(removelist)
    index = list(a_index - b_index)
    a = [list_name[i] for i in index]
    return a

#
def df_remove(dataframe, removelist, axis):
    if axis == 0:
        new_df = dataframe.drop(removelist)
        new_df = new_df.reset_index(drop=True)
    if axis == 1:
        new_df = dataframe.drop(removelist, axis=1)
        new_df.columns = range(new_df.shape[1])
    return new_df


#读取数据，药物的SMILES序列和蛋白质氨基酸序列
class LoadData(object):
    def __init__(self, setting_no, seqlen, smilen):

        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = Seq_Set
        self.charseqset_size = Seq_Set_len

        self.charsmiset = SMILES_Set  ###HERE CAN BE EDITED
        self.charsmiset_size = SMILES_Set_len
        self.PROBLEMSET = setting_no

    def parse_data(self, dataset_path, with_label=True):

        drugs = json.load(open(dataset_path + "drugs_iso.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(dataset_path + "proteins.txt"), object_pairs_hook=OrderedDict)
        drugs = orderdict_list(drugs)
        proteins = orderdict_list(proteins)
        
        if 'davis' in dataset_path:
            DTI = pd.read_csv(dataset_path + 'davis_dti.csv', sep='\s+', header=None, encoding='latin1')

        elif 'bindingdb' in dataset_path:
            DTI = pd.read_csv(dataset_path + 'BindingDB_Kd.txt', sep='\s+', header=None, encoding='latin1')

        else:
            DTI = pd.read_csv(dataset_path + 'kiba_dti.csv', sep='\s+', header=None, encoding='latin1')

        drugs_remove = get_removelist(drugs, 100)
        proteins_remove = get_removelist(proteins, 1200)
        drugs = list_remove(drugs, drugs_remove)
        proteins = list_remove(proteins, proteins_remove)
        DTI = df_remove(DTI, drugs_remove, 0)
        DTI = df_remove(DTI, proteins_remove, 1)

        XD = []
        XT = []
        if with_label:
            for d in drugs:
                XD.append(label_smiles(d, self.SMILEN, self.charsmiset))

            for t in proteins:
                XT.append(label_sequence(t, self.SEQLEN, self.charseqset))
        else:
            for d in drugs.keys():
                XD.append(one_hot_smiles(drugs[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))

        return XD, XT, np.array(DTI)
