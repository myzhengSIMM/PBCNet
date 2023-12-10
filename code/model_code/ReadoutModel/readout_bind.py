import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import dgl
import dgl.function as fn
from dgllife.model.readout import AttentiveFPReadout
from Final.final import Bind
from utilis.function import get_activation_func
from dgllife.model.gnn.gcn import GCN
from torch_sparse import SparseTensor
import functools
from dgl.nn.functional import edge_softmax


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='ReLU', bias=True):
        super(DenseLayer, self).__init__()
        if activation is not None:
            self.act = get_activation_func(activation)
        else:
            self.act = None
        if not bias:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, input_feat):
        if self.act is not None:
            return self.act(self.fc(input_feat))
        else:
            return self.fc(input_feat)



def distance(edge_feat):
    def func(edges):
        return {'dist': (edges.src[edge_feat]-edges.dst[edge_feat]).pow(2).sum(dim=-1).sqrt()}

    return func


def edge_cat(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: torch.cat([edges.data[src_field], edges.data[dst_field]],dim=-1)}

    return func


class DMPNN(nn.Module):
    """
    the whole model (PBCNet)
    """
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers, num_heads=4,
                 in_dim_atom=70, in_dim_edge=14, encoder_type="DMPNN_res", readout_type="AttFP",output_dim=1, degree_information=0, GCN_=0,
                 cs=0, two_task = 0):
        super(DMPNN, self).__init__()

        self.encoder_type = encoder_type
        self.readout_type = readout_type
        self.degree_information = degree_information
        self.GCN_ = GCN_
        self.cs = cs

        if encoder_type == "Bind":
            self.encoder = Bind(num_head=num_heads, feat_drop=p_dropout, attn_drop=p_dropout,
                                num_convs=radius, hidden_dim=hidden_dim, activation="ReLU")
        if readout_type == "AttFP":
            self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                          num_timesteps=T,
                                          dropout=p_dropout)


        self.hidden_dim = hidden_dim
        self.bias = False

        self.act_func = get_activation_func("ReLU")
        self.dropout = p_dropout
        self.num_FFN_layer = ffn_num_layers

        self.output_dim = output_dim

        ffn = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
        ffn.append(nn.Linear(hidden_dim * 2, hidden_dim))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
        ffn.append(nn.ReLU())
        ffn.append(nn.Linear(int(hidden_dim * 0.5), self.output_dim))

        self.FNN = nn.Sequential(*ffn)


        self.GCN = GCN(in_feats=70, hidden_feats=[hidden_dim, hidden_dim, hidden_dim],
                       batchnorm=[True, True, True])

        self.lin_atom1 = DenseLayer(in_dim=in_dim_atom, out_dim=hidden_dim, bias=True)
        self.lin_edge1 = DenseLayer(in_dim=in_dim_edge, out_dim=int(hidden_dim / 2), bias=True)

        self.lin1_cs = DenseLayer(hidden_dim, hidden_dim)
        self.lin2_cs = DenseLayer(hidden_dim, hidden_dim)

        if two_task == 1:

            ffn_2 = [nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU()]
            ffn_2.append(nn.Linear(hidden_dim * 2, hidden_dim))
            ffn_2.append(nn.ReLU())
            ffn_2.append(nn.Linear(hidden_dim, int(hidden_dim * 0.5)))
            ffn_2.append(nn.ReLU())
            ffn_2.append(nn.Linear(int(hidden_dim * 0.5),  2))   # high or low
            self.FNN2 = nn.Sequential(*ffn_2)
        self.two_task = two_task

    def triplets(self, g):
        row, col =  g.edges()  # j --> i
        num_nodes = g.num_nodes()

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, g1, g2, g_pocket=None):

        i1, j1, idx_i1, idx_j1, idx_k1, idx_kj1, idx_ji1 = self.triplets(g1)
        i2, j2, idx_i2, idx_j2, idx_k2, idx_kj2, idx_ji2 = self.triplets(g2)

        k1 = torch.nonzero(g1.ndata["p_or_l"], as_tuple=True)[0]  # 口袋原子索引
        k2 = torch.nonzero(g2.ndata["p_or_l"], as_tuple=True)[0]

        k1_ = torch.nonzero(g1.ndata["p_or_l"] == 0, as_tuple=True)[0]  # 配体原子索引
        k2_ = torch.nonzero(g2.ndata["p_or_l"] == 0, as_tuple=True)[0]

        if self.GCN_ == 1:   # 1=open
            g_pocket = dgl.add_self_loop(g_pocket)
            g_pocket.ndata["atom_feature_h"] = self.GCN(g_pocket, g_pocket.ndata["atom_feature"].to(torch.float32))

            g1.ndata["atom_feature_h"] = torch.zeros([g1.ndata["atom_feature"].shape[0], self.hidden_dim]).to(device=g1.device)
            g2.ndata["atom_feature_h"] = torch.zeros([g2.ndata["atom_feature"].shape[0], self.hidden_dim]).to(device=g1.device)

            g1.ndata["atom_feature_h"][k1_] = self.lin_atom1(
                g1.ndata["atom_feature"].to(torch.float32)[k1_])  # 70 --> 200
            g2.ndata["atom_feature_h"][k2_] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32)[k2_])

            g1.ndata["atom_feature_h"][k1] = g_pocket.ndata["atom_feature_h"]  
            g2.ndata["atom_feature_h"][k2] = g_pocket.ndata["atom_feature_h"]

        if self.GCN_ == 0:   # 0 = close
            g1.ndata["atom_feature_h"] = self.lin_atom1(g1.ndata["atom_feature"].to(torch.float32))  # 70 --> 200
            g2.ndata["atom_feature_h"] = self.lin_atom1(g2.ndata["atom_feature"].to(torch.float32))


        g1.edata["edge_feature_h"] = self.lin_edge1(g1.edata["edge_feature"].to(torch.float32))  # 14 --> 100
        g2.edata["edge_feature_h"] = self.lin_edge1(g2.edata["edge_feature"].to(torch.float32))

        g1.apply_edges(distance('atom_coordinate'))  # 获得dist
        g2.apply_edges(distance('atom_coordinate'))

        # attention bias (distance information)
        diss1 = torch.where(g1.edata['attention_weight'] == 0,
                            torch.tensor(-1).to(device=g1.device, dtype=torch.float32),
                            torch.log(g1.edata['attention_weight'])*2)

        diss2 = torch.where(g2.edata['attention_weight'] == 0,
                            torch.tensor(-1).to(device=g2.device, dtype=torch.float32),
                            torch.log(g2.edata['attention_weight'])*2)

        g1.edata['dist_decay'] = torch.where(g1.edata['attention_weight'] == 1,
                                             torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
                                             diss1)
        g2.edata['dist_decay'] = torch.where(g2.edata['attention_weight'] == 1,
                                             torch.tensor(1.0).to(device=g1.device, dtype=torch.float32),
                                             diss2)

        h1,att1 = self.encoder(g1, idx_kj1, idx_ji1, idx_i1, idx_j1, idx_k1)
        h2,att2 = self.encoder(g2, idx_kj2, idx_ji2, idx_i2, idx_j2, idx_k2)


        with g1.local_scope():
            with g2.local_scope():
                g1.ndata['h'] = h1
                g2.ndata['h'] = h2

                d1 = g1.in_degrees()[k1_].unsqueeze(dim=-1)
                d2 = g2.in_degrees()[k2_].unsqueeze(dim=-1)

                g1.remove_nodes(k1)
                g2.remove_nodes(k2)

                if self.readout_type == "AttFP":
                    hsg1 = self.readout(g1, g1.ndata['h'], False)
                    hsg2 = self.readout(g2, g2.ndata['h'], False)


                zk1 = self.FNN(
                    torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32)],
                              dim=-1))
                if self.two_task == 1:
                    zk2 = self.FNN2(torch.cat([hsg1.to(torch.float32), hsg2.to(torch.float32), (hsg1 - hsg2).to(torch.float32)],
                              dim=-1))
                    return zk1,zk2,att1,att2
                else:
                    return zk1,att1,att2

