# the code for the building of AU-MPNN

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax
from utilis.function import get_activation_func


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



class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_dim, act="ReLU"):
        super(ResidualLayer, self).__init__()
        self.act = get_activation_func(act)
        self.lin1 = DenseLayer(hidden_dim, hidden_dim,activation=act)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim,activation=act)

    def forward(self, he):
        return he + self.lin2(self.lin1(he))



class Distance2embedding(nn.Module):
    """
    Implementation of Spatial distance Embedding Module.
    """

    def __init__(self, hidden_dim, cut_dist, activation="ReLU"):
        super(Distance2embedding, self).__init__()

        self.cut_dist = cut_dist

        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1 , hidden_dim)  

        self.dist_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)  

    def forward(self, dist_feat):
        dist = torch.clamp(dist_feat.squeeze(), 1.0, self.cut_dist - 1e-6).type(
            torch.int64) - 1 

        distance_emb = self.dist_embedding_layer(dist)  

        distance_emb = self.dist_input_layer(distance_emb)  

        return distance_emb



class Angle2embedding(nn.Module):
    """
    Implementation of Spatial angle Embedding Module.
    """
    def __init__(self, hidden_dim, class_num, activation="ReLU"):

        super(Angle2embedding, self).__init__()

        self.class_num = class_num
        self.angle_embedding_layer = nn.Embedding(int(class_num), hidden_dim)  
        self.angle_input_layer = DenseLayer(hidden_dim, hidden_dim, activation, bias=True)  

    def forward(self, angle):

        angle = (angle/ (3.1415926/self.class_num)).type(torch.int64)
        angle_emb = self.angle_embedding_layer(angle) 
        angle_emb = self.angle_input_layer(angle_emb)

        return angle_emb



def src_cat_edge(src_field, edge_field, out_field):
    def func(edges):
        return {out_field: torch.cat((edges.src[src_field], edges.data[edge_field]), dim=-1)}

    return func


class Atom2BondLayer(nn.Module):
    """
    Initial hidden representations for edges.
    """
    def __init__(self, hidden_dim, activation="ReLU"):
        super(Atom2BondLayer, self).__init__()
        in_dim = int(hidden_dim * 1.5)
        self.lin1 = DenseLayer(in_dim, hidden_dim, activation=activation, bias=True)

    def forward(self, g, atom_embedding, edge_embedding):
        with g.local_scope():
            g.ndata['h'] = atom_embedding
            g.edata['h'] = edge_embedding
            g.apply_edges(src_cat_edge('h', 'h', 'h'))
            h = self.lin1(g.edata['h'])
        return h


class Bond2BondLayer(nn.Module):
    """
    Implementation of Distance and angle-aware edge to edge blocks (DAEE blocks).
    """
    def __init__(self, hidden_dim, num_head, feat_drop=0, attn_drop=0, activation="ReLU", activation_att="LeakyReLU",class_num=6,dist_cutoff=5):
        super(Bond2BondLayer, self).__init__()

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(hidden_dim / num_head)
        self.num_head = num_head
        self.hidden_dim = hidden_dim

        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_head, self.out_feat)))
        self.act_att = get_activation_func(activation_att)

        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=None)

        self.res1 = ResidualLayer(hidden_dim,activation)
        self.res2 = ResidualLayer(hidden_dim,activation)

        # angle information
        self.angle_embedding  = Angle2embedding(hidden_dim, class_num)
        self.angle1 = DenseLayer(hidden_dim, hidden_dim)
        self.angle2 = DenseLayer(hidden_dim, hidden_dim)
        self.act_func = get_activation_func(activation)


    def forward(self, graph, bond_embedding, index_kj, index_ji, idx_i,idx_j,idx_k):
        with graph.local_scope():

            pos = graph.ndata["atom_coordinate"]

            # Calculate angles.
            pos_i = pos[idx_i]
            pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
            a = (pos_ji * pos_ki).sum(dim=-1)
            b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
            angle = torch.atan2(b, a)     # [0,3.14]

            angle_embedding = self.angle_embedding(angle)
            angle_embedding = self.angle1(self.angle2(angle_embedding)).view(-1, self.num_head, self.out_feat)

            dist_decay = graph.edata["dist_decay"][index_kj]
            bond_embedding_feats = self.feat_drop(bond_embedding)

            feat_kj = self.k(bond_embedding_feats).view(-1, self.num_head, self.out_feat)[index_kj]
            feat_ji = self.q(bond_embedding_feats).view(-1, self.num_head, self.out_feat)[index_ji]

            feat = feat_kj + feat_ji + angle_embedding
            feat = self.act_att(feat)
            att = (feat * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # (num_edge_pair, num_heads, 1)

            att_decay = att + dist_decay.unsqueeze(dim=1)

            # soft max
            att_decay = torch.exp(att_decay)
            att_all = torch.zeros(len(bond_embedding), self.num_head, 1).to(bond_embedding.device)
            att_all = att_all.index_add_(0, index_ji, att_decay)
            att_all = att_all[index_ji]
            att_decay = self.attn_drop(att_decay / att_all)

            # residual
            v_att = (feat_kj * att_decay).view(-1, self.hidden_dim)
            v_clone = bond_embedding.clone()
            v_clone = v_clone.index_add_(0, index_ji, v_att) - v_clone
            he = self.lin2(self.lin1(v_clone))
            he = he + bond_embedding
            return self.res2(self.res1(he))


def e_mul_e(edge_field1, edge_field2, out_field):
    def func(edges):
        # clamp for softmax numerical stability
        return {out_field: edges.data[edge_field1] * edges.data[edge_field2]}

    return func


class Bond2AtomLayer(nn.Module):
    """
    Implementation of Distance-aware edge to node blocks (DEN blocks).
    """
    def __init__(self, hidden_dim, num_head, feat_drop=0, attn_drop=0, activation="ReLU", activation_att="LeakyReLU", dist_cutoff=5):
        super(Bond2AtomLayer, self).__init__()

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_feat = int(hidden_dim / num_head)
        self.num_head = num_head
        self.hidden_dim = hidden_dim

        self.k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_head, self.out_feat)))
        self.act_att = get_activation_func(activation_att)

        self.lin1 = DenseLayer(hidden_dim, hidden_dim, activation)
        self.lin2 = DenseLayer(hidden_dim, hidden_dim, activation=None)

        self.res1 = ResidualLayer(hidden_dim,activation)
        self.res2 = ResidualLayer(hidden_dim,activation)


    def forward(self, graph, bond_embedding, atom_embedding, att_para=False):
        with graph.local_scope():

            graph.edata['bond_embedding'] = bond_embedding.view(-1, self.num_head, self.out_feat)

            dist_decay = graph.edata["dist_decay"]
            atom_h = self.feat_drop(atom_embedding)

            graph.ndata["k"] = self.k(atom_h).view(-1, self.num_head, self.out_feat)
            graph.ndata["q"] = self.q(atom_h).view(-1, self.num_head, self.out_feat)
            graph.apply_edges(fn.u_add_v('k', 'q', 'e'))

            att = self.act_att(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            att = (att * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # (num_edge, num_heads, 1)
            att_decay = att + dist_decay.unsqueeze(dim=1) 
            att_decay = edge_softmax(graph, att_decay)

            graph.edata['att_decay'] = self.attn_drop(att_decay)
            graph.update_all(e_mul_e('bond_embedding', 'att_decay', 'm'),
                             fn.sum('m', 'ft'))

            he = graph.ndata['ft'].view(-1, self.hidden_dim)
            he = self.lin2(self.lin1(he))
            he = he + atom_embedding

            if att_para:
                return self.res2(self.res1(he)), att_decay
            else:
                return self.res2(self.res1(he))


class Bind(nn.Module):
    """
    Implementation of the whole AU-MPNN.
    """
    def __init__(self,  num_head=4, feat_drop=0, attn_drop=0, num_convs=4, hidden_dim=200,activation="ReLU"):
        super(Bind, self).__init__()

        self.num_convs = num_convs

        # self.a2b_layers = nn.ModuleList()
        self.b2b_layers = nn.ModuleList()
        self.b2a_layers = nn.ModuleList()

        # self.a2b_layers.append(Atom2BondLayer(hidden_dim, activation=activation))
        self.a2b = Atom2BondLayer(hidden_dim, activation=activation)

        for i in range(num_convs):
            self.b2b_layers.append(Bond2BondLayer(hidden_dim, num_head=num_head, feat_drop=feat_drop,
                                                  attn_drop=attn_drop, activation=activation))
            self.b2a_layers.append(Bond2AtomLayer(hidden_dim, num_head=num_head, feat_drop=feat_drop,
                                                  attn_drop=attn_drop, activation=activation))

    def forward(self, g, index_kj, index_ji,idx_i,idx_j,idx_k):
        bond_embedding = g.edata["edge_feature_h"]
        atom_embedding = g.ndata["atom_feature_h"]

        bond_embedding = self.a2b(g, atom_embedding, bond_embedding)
        for layer_num in range(self.num_convs-1):
            bond_embedding = self.b2b_layers[layer_num](g, bond_embedding, index_kj, index_ji, idx_i,idx_j,idx_k)
            atom_embedding = self.b2a_layers[layer_num](g, bond_embedding, atom_embedding)

        bond_embedding = self.b2b_layers[-1](g, bond_embedding, index_kj, index_ji, idx_i,idx_j,idx_k)
        atom_embedding,att = self.b2a_layers[-1](g, bond_embedding, atom_embedding,att_para=True)

        return atom_embedding,att
