# define dataloader and collec function
import os
import dgl
import numpy as np
import pandas as pd
import torch

from utilis.utilis import Extend, pkl_load

code_path = os.path.dirname(os.path.abspath(__file__))
code_path = code_path.split('code')[0] + "data"


def collate_fn(samples):

    ligand1_dir = [code_path + s.Ligand1.values[0].split("data")[-1] for s in samples]
    ligand2_dir = [code_path + s.Ligand2.values[0].split("data")[-1] for s in samples]
    pocket_dir = [code_path + s.Ligand2.values[0].split("data")[-1].rsplit("/", 1)[0] + "/pocket.pkl" for s in samples]
    graph1_list = [pkl_load(s) for s in ligand1_dir]
    graph2_list = [pkl_load(s) for s in ligand2_dir]
    pocket_list = [pkl_load(s) for s in pocket_dir]

    g1 = dgl.batch(graph1_list)
    g2 = dgl.batch(graph2_list)
    pock = dgl.batch(pocket_list)
    # index_kj1, index_ji1 = triplets(g1)
    # index_kj2, index_ji2 = triplets(g2)

    label_list = [s.Lable.values[0] for s in samples]  # delta value (True label)
    label1_list = [s.Lable1.values[0] for s in samples]  # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]  # referance train samples' labels

    rank1_list = [s.Rank1.values[0] for s in samples]  
    file_name = [s.rsplit("/", 2)[1] for s in ligand1_dir]

    return g1, \
           g2, \
           pock, \
           torch.tensor(label_list), \
           torch.tensor(label1_list), \
           torch.tensor(label2_list), \
           torch.tensor(rank1_list), \
           file_name


class LeadOptDataset():
    def __init__(self, df_path, label_scalar=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar == "finetune":
            label = self.df.Lable.values
            label = (np.array(label).astype(float) - 0.04191832) / 1.34086546
            self.df["Lable"] = label

        elif self.label_scalar is not None:
            label = self.df.Lable.values
            label = np.reshape(label, (-1, 1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        # self.df = self.df[0:256]
        super(LeadOptDataset, self).__init__()

            
    def file_names_(self):
        ligand_dir = self.df.Ligand1.values
        file_names = [s.rsplit("/", 2)[1] for s in ligand_dir]
        return list(set(file_names))

        
    def __getitem__(self, idx):
        return self.df[idx:idx + 1]

    def __len__(self):
        return len(self.df)

# for finetune
class LeadOptDataset_retrain():
    def __init__(self, df_path, corr_path, avoid_forget=0):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)

        corr = pd.read_csv(corr_path)
        corr_small = corr[corr.spearman <= 0.5].file_name.values

        self.df["file_name"] = [i.rsplit("/",2)[1] for i in self.df.Ligand1.values]

        self.df_new = self.df[self.df["file_name"].isin(corr_small)]

        if avoid_forget == 1:
            self.df_good_part = self.df[~self.df["file_name"].isin(corr_small)]
            self.df_good_part = self.df_good_part.sample(n=len(self.df_new), replace=False, random_state=2)
            self.df_new = pd.concat([self.df_new,self.df_good_part], ignore_index=True)
            
        super(LeadOptDataset_retrain, self).__init__()
        
    def __getitem__(self, idx):
        return self.df_new[idx:idx + 1]

    def __len__(self):
        return len(self.df_new)


class LeadOptDataset_test():
    def __init__(self, df_path, label_scalar=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar == "finetune":
            label = self.df.Lable.values
            label = (np.array(label).astype(float) - 0.04191832) / 1.34086546
            self.df["Lable"] = label

        elif self.label_scalar is not None:
            label = self.df.Lable.values
            label = np.reshape(label, (-1, 1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        self.df = self.df[0:256]
        super(LeadOptDataset_test, self).__init__()

    def file_names_(self):
        ligand_dir = self.df.Ligand1.values
        file_names = [s.rsplit("/", 2)[1] for s in ligand_dir]
        return list(set(file_names))

    def __getitem__(self, idx):
        return self.df[idx:idx + 1]

    def __len__(self):
        return len(self.df)