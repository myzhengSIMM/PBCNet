# import package
import dgl
from dgl.data import DGLDataset
import dgl.function as fn
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn

import dgllife
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import argparse
import math

import os
import sys
sys.path.append("/home/yujie/AIcode/utilis/")

from trick import Writer
from scalar import StandardScaler
from initial import initialize_weights
from function import get_loss_func, get_activation_func
from scheduler import NoamLR_shan

cpu_num = 4
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def get_parser():
    parser = argparse.ArgumentParser(
        "LeadOpt Model"
    )
    parser.add_argument(
        '--scalar', type=bool, default=True, help='Whether to normalize the labels of training set.'
        )
    parser.add_argument(
        '--loss_func', type=str, default="smoothl1", help='The loss function used to train the model: mse, smoothl1, mve, evidential'
        )
    parser.add_argument(
        "--device", type=int, default=0, help="The number of device: 0,1,2,3 [on v100-2]"
        )
    parser.add_argument(
        '--continue_learning',type=bool,default=True,help='Whether to use continue learning in the training process'
        )
    # parser.add_argument(
    #     '--scheduler',type=bool, default=True, help='Whether to adjust the learning rate of model by a scheduler.'
    #     )

    return parser

def pkl_load(file_name):
    pickle_file = open(file_name,'rb')
    graph,embedding,atomnum_of_ligand = pickle.load(pickle_file)
    return graph,embedding,atomnum_of_ligand

def pkl_load_gm(file_name):
    pickle_file = open(file_name,'rb')
    a = pickle.load(pickle_file)
    return a

def Extend(list1,list2):
    list1.extend(list2)
    return list1

def gm_to_batch(gm,lenth):
    for x in range(len(gm)):
        if x == 0:
            continue
        gm[x] = gm[x] + lenth[x-1]
        
    gm = torch.cat(gm,dim = 0)
    return gm

def gm_process(path_list, graph,device):
    dist = []
    angle = []
    torsion = []
    i = []
    j = []
    idx_kj = [] 
    idx_ji = []
    incomebond_edge_ids = []
    incomebond_index_to_atom = []
    for s in path_list:
        dist_, angle_, torsion_, i_, j_, idx_kj_, idx_ji_, incomebond_edge_ids_, incomebond_index_to_atom_ = pkl_load_gm(s.rsplit(".",1)[0] + "_gm.pkl")
        dist.append(dist_)
        angle.append(angle_)
        torsion.append(torsion_)
        i.append(i_)
        j.append(j_)
        idx_kj.append(idx_kj_) 
        idx_ji.append(idx_ji_)
        incomebond_edge_ids.append(incomebond_edge_ids_)
        incomebond_index_to_atom.append(incomebond_index_to_atom_)
    dist = torch.cat(dist,dim = 0)
    angle = torch.cat(angle,dim = 0)
    torsion = torch.cat(torsion,dim = 0)

    n_nodes = graph.batch_num_nodes()
    n_nodes = torch.tensor([torch.sum(n_nodes[0:i+1]) for i in range(len(n_nodes))])

    n_edges = graph.batch_num_edges()
    n_edges = torch.tensor([torch.sum(n_edges[0:i+1]) for i in range(len(n_edges))])

    i = gm_to_batch(i,n_nodes)
    j = gm_to_batch(j,n_nodes)
    idx_kj = gm_to_batch(idx_kj,n_edges)
    idx_ji = gm_to_batch(idx_ji,n_edges)
    incomebond_edge_ids = gm_to_batch(incomebond_edge_ids,n_edges)
    incomebond_index_to_atom = gm_to_batch(incomebond_index_to_atom,n_nodes)
    
    return dist.to(device), angle.to(device), torsion.to(device), i.to(device), j.to(device), \
           idx_kj.to(device), idx_ji.to(device), incomebond_edge_ids.to(device), incomebond_index_to_atom.to(device)

# define dataloader and collec function
def collate_fn(samples):

    ligand1_dir = [s.Ligand1.values[0] for s in samples]
    ligand2_dir = [s.Ligand2.values[0] for s in samples]
    graph1_list = [pkl_load(s)[0] for s in ligand1_dir]
    graph2_list = [pkl_load(s)[0] for s in ligand2_dir]

    # gm1_list = [pkl_load_gm(s.rsplit(".",1)[0] + "_gm.pkl") for s in ligand1_dir]
    # gm2_list = [pkl_load_gm(s.rsplit(".",1)[0] + "_gm.pkl") for s in ligand2_dir]

    ligand1_atomnum_list = [list(range(pkl_load(s)[2])) for s in ligand1_dir]
    ligand1_padding = [len(i) for i in ligand1_atomnum_list]                                             # 一维 记录长度 [3,1]
    ligand1_num = max(ligand1_padding)
    ligand1_atomnum_list = [ Extend(i, list(range(ligand1_num-len(i)))) for i in ligand1_atomnum_list]   # 二维 [[1,3,2],[0,0,1]]

    ligand2_atomnum_list = [list(range(pkl_load(s)[2])) for s in ligand2_dir]
    ligand2_padding = [len(i) for i in ligand2_atomnum_list]
    ligand2_num = max(ligand2_padding)
    ligand2_atomnum_list = [ Extend(i, list(range(ligand2_num-len(i)))) for i in ligand2_atomnum_list]

    label_list = [s.Lable.values[0] for s in samples]  # delta
    label1_list = [s.Lable1.values[0] for s in samples]   # validation samples' labels
    label2_list = [s.Lable2.values[0] for s in samples]   # referance train samples' labels
                                                          # preditced pIC50 = predicted delta + referance train samples' labels
                                                          # RMSD = (preditced pIC50, validation samples' labels)
    rank1_list = [s.Rank1.values[0] for s in samples]   # 用于识别pair属于哪一个validation sample

    file_name = [s.rsplit("/",2)[1] for s in ligand1_dir]

    interaction_embedding1 = [pkl_load(s)[1] for s in ligand1_dir] 
    interaction_embedding2 = [pkl_load(s)[1] for s in ligand2_dir]

    return dgl.batch(graph1_list), \
           dgl.batch(graph2_list), \
           torch.tensor(label_list),\
           torch.tensor(ligand1_atomnum_list),\
           torch.tensor(ligand2_atomnum_list),\
           torch.tensor(ligand1_padding),\
           torch.tensor(ligand2_padding),\
           torch.tensor(interaction_embedding1),\
           torch.tensor(interaction_embedding2),\
           torch.tensor(label1_list),\
           torch.tensor(label2_list),\
           torch.tensor(rank1_list),\
           file_name,\
           ligand1_dir,\
           ligand2_dir

class LeadOptDataset():
    def __init__(self, df_path, label_scalar = None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.label_scalar = label_scalar

        if self.label_scalar != None:
            label = self.df.Lable.values
            label = np.reshape(label,(-1,1))
            self.label_scalar = self.label_scalar.fit(label)
            label = self.label_scalar.transform(label)
            self.df["Lable"] = label.flatten()

        # self.df = self.df[0:256]
        super(LeadOptDataset, self).__init__()

    def __getitem__(self, idx):
        return self.df[idx:idx+1]

    def __len__(self):
        return len(self.df)

# define model
class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act="ReLU",p_dropout=0.8):
        super(ResidualLayer, self).__init__()
        self.act = get_activation_func(act)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = p_dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # self.reset_parameters()
    # def reset_parameters(self):
    #     glorot_orthogonal(self.lin1.weight, scale=2.0)
    #     self.lin1.bias.data.fill_(0)
    #     glorot_orthogonal(self.lin2.weight, scale=2.0)
    #     self.lin2.bias.data.fill_(0)
    def forward(self, message, message_clone):
        return message + self.dropout_layer(self.act(self.lin2(self.act(self.lin1(message_clone)))))

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, _num_heads, bias = False):
        super(AttentionLayer,self).__init__()
        self.hidden_dim = hidden_dim
        self._num_heads = _num_heads
        self._out_feats = int(hidden_dim / _num_heads)
        self.bias = bias
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

        self.W_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.W_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)
        self.W_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)

        self.res = ResidualLayer(self.hidden_dim)
    
    def forward(self, feats, idx_kj, idx_ji):
        v = self.W_v(feats).view(-1, self._num_heads, self._out_feats)  # num_bonds x head x features
        k = self.W_k(feats).view(-1, self._num_heads, self._out_feats)  # num_bonds x head x features
        q = self.W_q(feats).view(-1, self._num_heads, self._out_feats)  # num_bonds x head x features

        q_kj = q[idx_kj]
        k_ji = k[idx_ji]
        att = (q_kj * k_ji).sum(dim = -1).unsqueeze(-1)
        att = self.leaky_relu(att)

        att = torch.exp(att)
        # soft max
        att_all = torch.zeros(len(feats), self._num_heads,1).to(feats.device)
        att_all = att_all.index_add_(0, idx_ji, att)
        att_all = att_all[idx_ji]
        att = att / att_all

        v_att = (v[idx_kj] * att).view(len(idx_kj), self.hidden_dim)
        v = v.view(-1, self.hidden_dim)
        v_clone = v.clone()
        v_clone = v_clone.index_add_(0,idx_ji,v_att) - v_clone

        v = self.res(v,v_clone)
        return v


class DMPNN_Encoder(nn.Module):
    def __init__(self, hidden_dim, radius, p_dropout):
        super(DMPNN_Encoder, self).__init__()

        self.atom_feature_dim = 133
        self.bond_feature_dim = 14
        self.hidden_dim = hidden_dim
        self._num_heads = 8
        self.bias = False
        self.input_dim = 133+14

        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.num_MPNN_layer =  radius
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.W_i = nn.Linear(self.input_dim, self.hidden_dim, bias=self.bias)
        self.Att_layer = torch.nn.ModuleList([
                                              AttentionLayer(self.hidden_dim,self._num_heads)
                                              for _ in range(self.num_MPNN_layer-1)
                                              ])

        self.W_o = nn.Linear(self.atom_feature_dim + self.hidden_dim, self.hidden_dim)


    def forward(self, G, gm):

        dist, angle, torsion, i, j, idx_kj, idx_ji, incomebond_edge_ids, incomebond_index_to_atom = gm
        num_bonds = G.num_edges()

        G.edata["edge_feature"] = G.edata["edge_feature"].float()
        def initial_func(edges): # fea(j)+fea(j-->i) num_bonds x (133+14)
            return {'initial_feature': torch.cat((edges.src["atom_feature"],edges.data["edge_feature"]),dim=1)}
        G.apply_edges(initial_func)

        feats = G.edata["initial_feature"]
        feats = self.act_func(self.W_i(feats))  # num_bonds x hidden_size

        for layer in self.Att_layer:
            feats = layer(feats, idx_kj, idx_ji)

        # 把incoming bond的信息都归到原子上
        G.edata["feats"] = feats
        G.update_all(fn.copy_e('feats', 'm'), fn.sum('m', 'feats_sum'))
        # 融入原子本身的信息
        a_input = torch.cat([G.ndata["atom_feature"], G.ndata["feats_sum"]], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        return atom_hiddens


class DMPNN(nn.Module):
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers, output_dim = 1):
        
        super(DMPNN, self).__init__()

        self.encoder = DMPNN_Encoder(hidden_dim = hidden_dim, radius = radius, p_dropout = p_dropout)

        self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                          num_timesteps=T,
                                          dropout=p_dropout)
        self.hidden_dim = hidden_dim
        self.bias = False

        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.num_FFN_layer = ffn_num_layers

        self.output_dim = output_dim

        ffn = [nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU()]
        for _ in range(ffn_num_layers-1):
            ffn.append(nn.Linear(hidden_dim, hidden_dim))
            ffn.append(nn.ReLU())
        ffn.append(nn.Linear(hidden_dim, self.output_dim))

        self.FNN =nn.Sequential(*ffn)


    def forward(self, graph1, alist1, num1, graph2, alist2, num2, gm1, gm2):

        h1 = self.encoder(graph1, gm1)
        h2 = self.encoder(graph2, gm2)
        
        with graph1.local_scope():
            with graph2.local_scope():
                graph1.ndata['h'] = h1
                graph2.ndata['h'] = h2

                ug1 = dgl.unbatch(graph1)
                ug2 = dgl.unbatch(graph2) 

                sg1 = dgl.batch([ dgl.node_subgraph(ug1[i], alist1[i][0:num1[i]]) for i in range(len(ug1)) ])
                sg2 = dgl.batch([ dgl.node_subgraph(ug2[i], alist2[i][0:num2[i]]) for i in range(len(ug2)) ])

                # dgl.node_subgraph(g1, num1 )
                # sg2 = dgl.node_subgraph(g2, num2 )

                hsg1 = self.readout(sg1, sg1.ndata['h'], False)
                hsg2 = self.readout(sg2, sg2.ndata['h'], False)
                zk = self.FNN(torch.cat([hsg1, hsg2, hsg1-hsg2], dim=-1))
        return zk




def func(device = 'cuda:0', label_scalar = None, loss_fn = "smoothl1", continue_learning = True):

    if label_scalar == None:
        SAVEDIR = os.path.join(os.environ["HOME"],"leadopt", "results", f"DMPNNatt_pair_{loss_fn}_scalarFalse_continue{continue_learning}")
    else:
        SAVEDIR = os.path.join(os.environ["HOME"],"leadopt", "results", f"DMPNNatt_pair_{loss_fn}_scalarTrue_continue{continue_learning}")

    logger_writer = Writer(os.path.join(SAVEDIR, "record.txt"))

    # for key in INT_KEYS:
    #     hyperparams[key] = int(hyperparams[key])

    # for k in hyperparams:
    #     logger_writer(f"{k}\t{hyperparams[k]}")
    #     logger_writer(" ")

    validation_roc_list  = []



    train_dataset = LeadOptDataset("/home/yujie/leadopt/data/ic50_graph_rmH/trapart_insplit_07.csv", label_scalar)

    label_scalar = train_dataset.label_scalar

    # pickle_file_ = open("/home/yujie/leadopt/results/scalar.pkl",'wb')
    # pickle.dump(label_scalar,pickle_file_)
    # pickle_file_.close()

    valid_dataset = LeadOptDataset("/home/yujie/leadopt/data/ic50_graph_rmH/valpart_insplit_07.csv")
    # test_dataset = CoCrystalDataset("/home/wangdingyan/data/five_fold_data/test.csv")

    train_dataloader = GraphDataLoader(train_dataset, collate_fn=collate_fn, batch_size=80, drop_last=False, shuffle= (not continue_learning),num_workers=4,pin_memory = True)
    valid_dataloader = GraphDataLoader(valid_dataset, collate_fn=collate_fn, batch_size=80, drop_last=False, shuffle=True,num_workers=4,pin_memory = True)
    # test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=128, drop_last=False,shuffle=False)
    # test_dataloader.smiles1, test_dataloader.smiles2 = test_dataset.smiles1_list, test_dataset.smiles2_list

    if loss_fn == 'mve':
        model = DMPNN(  hidden_dim     = 200,
                        radius         = 4,
                        T              = 3,
                        p_dropout      =  0.9,
                        ffn_num_layers = 3,
                        output_dim     = 2).to(device)
    elif loss_fn == 'evidential':
        model = DMPNN(  hidden_dim     = 200,
                        radius         = 4,
                        T              = 3,
                        p_dropout      =  0.9,
                        ffn_num_layers = 3,
                        output_dim     = 4).to(device)
    else:
        model = DMPNN(  hidden_dim     = 200,
                        radius         = 4,
                        T              = 3,
                        p_dropout      =  0.9,
                        ffn_num_layers = 3,
                        output_dim     = 1).to(device)
    initialize_weights(model)
    
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, eps = 1e-4)
    loss_func = get_loss_func(loss_fn)
    scheduler = NoamLR_shan(opt,
                            warmup_epochs = [0.5],
                            decay_epochs = [14],
                            final_epochs = [24],
                            steps_per_epoch = 5814,
                            init_lr = [0.0001],
                            max_lr = [0.001],
                            final_lr = [0.0001])
    
    stop_metric = 0
    not_change_epoch = 0
    Path = os.path.join(SAVEDIR, "model.pth")

    EPOCH = 0
    BATCH = 0 
    for epoch in range(9999):
        EPOCH += 1
        model.train()
        loss_all_train = []
        loss_for_ten_batch = []

        for k in train_dataloader:
            graph1 = k[0]
            graph2 = k[1]
            path_gm1 = k[-2]
            path_gm2 = k[-1]
            
            gm1 = gm_process(path_gm1, graph1, device)
            gm2 = gm_process(path_gm2, graph2, device)

            graph1, graph2, label, ligand1_atom_num, liagnd2_atom_num, ligand1_padding, ligand2_padding, \
            interaction_embedding1, interaction_embedding2, label1, label2, rank1, file_name = k[:-2]

            graph1, graph2, label, ligand1_atom_num, liagnd2_atom_num, ligand1_padding, ligand2_padding, \
            interaction_embedding1, interaction_embedding2, label1, label2, rank1, file_name = \
            graph1.to(device), graph2.to(device), label.to(device), ligand1_atom_num.to(device),\
            liagnd2_atom_num.to(device), ligand1_padding.to(device), ligand2_padding.to(device), \
            interaction_embedding1, interaction_embedding2, label1.to(device), \
            label2.to(device), rank1, file_name 

            
            BATCH += 1
            logits = model(graph1,
                           ligand1_atom_num,
                           ligand1_padding,
                           graph2,
                           liagnd2_atom_num,
                           ligand2_padding,
                           gm1,
                           gm2)

            if loss_fn == 'mve':
                loss = loss_func(logits[:,0].float(), label.to(device).float(), torch.exp(logits[:,1]).float())
            elif loss_fn == "evidential":
                loss = loss_func(logits[:,0].float(), \
                                 F.softplus(logits[:,1].float()), \
                                 F.softplus(logits[:,2].float())+1.0, \
                                 F.softplus(logits[:,3].float()), \
                                 label.to(device).float())
            else:
                loss = loss_func(logits.squeeze().float(), label.to(device).float())

            loss_for_ten_batch.append(loss.detach().cpu().item())
            # 有时候预测会有None或者inf，用0代替数组中的nan，用有限值代替inf
            predictions = np.array(logits[:,0].detach().cpu(),float)
            predictions  = np.nan_to_num(predictions)
            mae_ = mean_absolute_error(label.float().detach().cpu(), predictions)
            loss_all_train.append(label_scalar.inverse_transform(mae_))
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            if len(loss_for_ten_batch) == 100:
                l = np.mean(loss_for_ten_batch)
                print(f"Epoch {EPOCH} Batch {BATCH}  Loss {l}")
                logger_writer(f"Epoch {EPOCH} Batch {BATCH}  Loss {l}")
                print(f"Epoch {EPOCH} Batch {BATCH}  Learning rate {scheduler.get_lr()[0]}")
                logger_writer(f"Epoch {EPOCH} Batch {BATCH}  Learning rate {scheduler.get_lr()[0]}")
                loss_for_ten_batch = []

        model.eval()
        with torch.no_grad():
            
            valid_prediction = []
            valid_labels = []
            valid_1_labels = []
            ref_2_labels = []
            rank = []
            file = []

            for k in valid_dataloader:
                graph1 = k[0]
                graph2 = k[1]
                path_gm1 = k[-2]
                path_gm2 = k[-1]
            
                gm1 = gm_process(path_gm1, graph1, device)
                gm2 = gm_process(path_gm2, graph2, device)

                graph1, graph2, label, ligand1_atom_num, liagnd2_atom_num, ligand1_padding, ligand2_padding, \
                interaction_embedding1, interaction_embedding2, label1, label2, rank1, file_name = k[:-2]

                graph1, graph2, label, ligand1_atom_num, liagnd2_atom_num, ligand1_padding, ligand2_padding, \
                interaction_embedding1, interaction_embedding2, label1, label2, rank1, file_name = \
                graph1.to(device), graph2.to(device), label.to(device), ligand1_atom_num.to(device),\
                liagnd2_atom_num.to(device), ligand1_padding.to(device), ligand2_padding.to(device), \
                interaction_embedding1, interaction_embedding2, label1.to(device), \
                label2.to(device), rank1, file_name 

                logits = model(graph1,
                               ligand1_atom_num,
                               ligand1_padding,
                               graph2,
                               liagnd2_atom_num,
                               ligand2_padding,
                               gm1,
                               gm2)

                if loss_fn == 'mve':
                    valid_prediction.extend(logits[:,0].unsqueeze(-1).tolist())
                elif loss_fn == "evidential":
                    valid_prediction.extend(logits[:,0].unsqueeze(-1).tolist())
                else:
                    valid_prediction.extend(logits.tolist())

                valid_labels.extend(label.tolist())
                valid_1_labels.extend(label1.tolist())
                ref_2_labels.extend(label2.tolist())
                rank.extend(rank1.tolist())
                file.extend(file_name)

            if label_scalar == None:
                pass
            else:
                valid_prediction = label_scalar.inverse_transform(valid_prediction)


            mae = mean_absolute_error(valid_labels, valid_prediction)
            mse = mean_squared_error(valid_labels, valid_prediction)

            pre_abs_pic50 = np.array(valid_prediction).flatten() + np.array(ref_2_labels)

            # abs_mse = mean_squared_error(valid_1_labels, list(pre_abs_pic50))

            
            file_to_p = defaultdict(list)
            for pre, lab, r, f in zip(pre_abs_pic50,valid_1_labels,rank , file):
                file_to_p[f].append([pre, lab, r])
            
            spearman = []
            pearson = []
            lenth = []
            files_ = []
            pre_abs_pic50_mean = []
            valid_1_labels_mean = []

            for f in file_to_p.keys():
                _df = pd.DataFrame(file_to_p[f], columns = ['a','b','c'])
                _df = _df.groupby('c')[['a','b']].mean().reset_index()

                pre_abs_pic50_mean.extend(list(_df['a'].values))
                valid_1_labels_mean.extend(list(_df['b'].values))
                


                if len(_df) >= 5:
                    files_.append(f)
                    lenth.append(len(_df))

                    spear = _df[["a","b"]].corr(method='spearman')
                    spearman.append(spear.iloc[0,1])

                    pear = _df[["a","b"]].corr(method='pearson')
                    pearson.append(pear.iloc[0,1])


            abs_mse = mean_squared_error(pre_abs_pic50_mean, valid_1_labels_mean)


            csv_save_dir = os.path.join(SAVEDIR, f"2.3_scalar_corr_{EPOCH}.csv")
            corr_df = pd.DataFrame({'file_name':files_,\
                                    'spearman':spearman,\
                                    'pearson':pearson,\
                                    'num_of_val_data':lenth})
            corr_df.to_csv(csv_save_dir, index = 0)

            loss_all_train = np.array(loss_all_train,float)
            logger_writer("  ")
            logger_writer(f"Epoch {EPOCH}")
            logger_writer(f"Validation Set mae {mae}")
            logger_writer(f"Validation Set mse {mse}")
            logger_writer(f"Train Set mae {np.mean(loss_all_train)}")
            logger_writer(f"Validation Set absolute mse {abs_mse}")
            logger_writer(f"Validation Set spearman {np.mean(spearman)}")
            logger_writer(f"Validation Set pearson {np.mean(pearson)}")
            logger_writer(f"Validation Set spearman {np.nanmean(spearman)}")
            logger_writer(f"Validation Set pearson {np.nanmean(pearson)}")


            # print(f"Epoch {epoch}")
            print(f"Epoch {EPOCH} Validation Set mae {mae}")
            print(f"Epoch {EPOCH} Validation Set mse {mse}")
            print(f"Validation Set absolute mse {abs_mse}")
            print(f"Train Set mse {np.nanmean(loss_all_train)}")

            print(f"Validation Set spearman {np.mean(spearman)}")
            print(f"Validation Set pearson {np.mean(pearson)}")
            print(f"Validation Set spearman {np.nanmean(spearman)}")
            print(f"Validation Set pearson {np.nanmean(pearson)}")
  


            loss_all_train = []



            if np.nanmean(pearson) > stop_metric:
                stop_metric = np.nanmean(pearson)
                not_change_epoch = 0
                torch.save(model, Path)
                # test_prediction = []
                # test_labels = []
                # for k in test_dataloader:
                #     model.eval()
                #     logits = model(k[0].to("cuda:0"),
                #                    k[0].ndata["atomic"].to("cuda:0"),
                #                    k[0].edata["type"].to("cuda:0"),
                #                    k[1].to("cuda:0"),
                #                    k[1].ndata["atomic"].to("cuda:0"),
                #                    k[1].edata["type"].to("cuda:0"))

                #     test_prediction.extend(F.softmax(logits, dim=-1)[:, 1].tolist())
                #     test_labels.extend(k[2].tolist())
                # test_df = pd.DataFrame({"smiles_0": test_dataloader.smiles1,
                #                         "smiles_1": test_dataloader.smiles2,
                #                         "prediction": test_prediction,
                #                         "label": test_labels})
                # test_df.to_csv(os.path.join(BASESAVEDIR, f"test_{i}.csv"), index=False)

            else:
                not_change_epoch += 1
                logger_writer(f"Stop metric not change for {not_change_epoch}")
                logger_writer(f"Best Validation pearson {stop_metric}")

            if not_change_epoch >= 10 :
                validation_roc_list.append(stop_metric)
                logger_writer("Stop Training")

                return None
    # torch.save(model, Path)
    





    # logger(f"Validation mean auROC {np.mean(validation_roc_list)}")
    # test_prediciton_list = []
    # for i in [1, 2, 3, 4, 5]:
    #     df = pd.read_csv(os.path.join(BASESAVEDIR, f"test_{i}.csv"))
    #     test_prediciton_list.append(df["prediction"].to_numpy())
    #     test_label = df["label"].to_numpy()
    # test_prediction = np.mean(test_prediciton_list, axis=0)
    # logger(f"Test auROC       {roc_auc_score(test_label, test_prediction)}")
    # logger(f"Test ACC         {accuracy_score(test_label, test_prediction>0.5)}")
    # logger(f"Test Precision   {precision_score(test_label, test_prediction>0.5)}")
    # logger(f"Test Recall      {recall_score(test_label, test_prediction>0.5)}")
    # logger(f"Test F1          {f1_score(test_label, test_prediction>0.5)}")

    # return -np.mean(validation_roc_list)






if __name__ == "__main__":

    parser = get_parser()
    config, unknown = parser.parse_known_args()

    use_scalar = config.scalar
    loss_function = str(config.loss_func)
    continue_learning = config.continue_learning
    cuda = "cuda:" + str(config.device)


    if use_scalar == True:
        scalar = StandardScaler()
    else:
        scalar = None

    func(device=cuda, label_scalar = scalar, loss_fn=loss_function, continue_learning = continue_learning)








