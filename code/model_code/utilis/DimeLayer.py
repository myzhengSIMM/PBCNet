import torch
import torch.nn as nn
import dgl.function as fn
import math

import sys

sys.path.append("/home/yujie/AIcode/utilis/")
from function import get_activation_func
from DAencoder import BesselBasisLayer, SphericalBasisLayer


def glorot_orthogonal(tensor, scale):
    # 参数初始化方法
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class EmbeddingBlock(torch.nn.Module):
    # 通过原子类型和两个原子之间的距离来获得表征
    # return： [num_of_bond, hidden_channels]
    def __init__(self, num_radial, hidden_channels, act):
        """ num_radial: the number of rbf(径向基函数) to describe distance.
            hidden_channels: the dimension of latent space.
            act: the activation function.
        """
        super(EmbeddingBlock, self).__init__()
        self.act = get_activation_func(act)

        self.atom_type_emb = nn.Embedding(100, hidden_channels)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     self.atom_type_emb.weight.data.uniform_(-math.sqrt(3), math.sqrt(3))
    #     self.lin_rbf.reset_parameters()
    #     self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        """
        x: 原子的类型
        rbf: 距离表征
        i: 每条边的终点原子的索引
        j: 每条边的起始原子的索引
        """
        x = self.atom_type_emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    # 残差网络
    # 输入与输出相同
    def __init__(self, hidden_channels, act):
        super(ResidualLayer, self).__init__()
        self.act = get_activation_func(act)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     glorot_orthogonal(self.lin1.weight, scale=2.0)
    #     self.lin1.bias.data.fill_(0)
    #     glorot_orthogonal(self.lin2.weight, scale=2.0)
    #     self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_bilinear, num_spherical,
                 num_radial, num_before_skip, num_after_skip, act):
        super(InteractionBlock, self).__init__()
        self.act = get_activation_func(act)
        self.num_bilinear = num_bilinear

        # self.lin_rbf1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_rbf2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, num_bilinear,
                                  bias=False)
        self.lin_sbf2 = nn.Linear(num_bilinear, num_bilinear, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.down_projection = nn.Linear(hidden_channels, num_bilinear)
        self.up_projection = nn.Linear(num_bilinear, hidden_channels)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        # self.lin = nn.Linear(hidden_channels, hidden_channels)
        # self.layers_after_skip = torch.nn.ModuleList([
        #     ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        # ])

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
    #     glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
    #     glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
    #     glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
    #     glorot_orthogonal(self.down_projection.weight, scale=2.0)
    #     self.down_projection.bias.data.fill_(0)
    #     glorot_orthogonal(self.up_projection.weight, scale=2.0)
    #     self.up_projection.bias.data.fill_(0)
    #     glorot_orthogonal(self.lin_kj.weight, scale=2.0)
    #     self.lin_kj.bias.data.fill_(0)
    #     glorot_orthogonal(self.lin_ji.weight, scale=2.0)
    #     self.lin_ji.bias.data.fill_(0)
    #     # self.W.data.normal_(mean=0, std=2 / self.W.size(0))
    #     for res_layer in self.layers_before_skip:
    #         res_layer.reset_parameters()
    #     glorot_orthogonal(self.lin.weight, scale=2.0)
    #     self.lin.bias.data.fill_(0)
    #     for res_layer in self.layers_after_skip:
    #         res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        """
        x: 每一条变的初始表征（距离embedding和化学先验embedding一起）
        """
        # x_ji = self.act(self.lin_ji(x))
        x_ji = x
        x_kj = self.act(self.lin_kj(x))

        # 编码距离
        # rbf = self.lin_rbf1(rbf)
        rbf = self.act(self.lin_rbf2(rbf))
        x_kj = x_kj * rbf
        # x_kj = x_kj[idx_kj] * rbf[idx_ji]

        # 降维至角度信息
        x_kj = self.act(self.down_projection(x_kj))
        x_kj = x_kj[idx_kj]

        # 编码角度
        sbf = self.act(self.lin_sbf1(sbf))
        sbf = self.act(self.lin_sbf2(sbf))
        x_kj = x_kj * sbf

        # 综合邻居边的信息
        x_kj = self.act(self.up_projection(x_kj))
        message = torch.zeros(len(x), self.num_bilinear).to(x_kj.device)
        x_kj = message.index_add_(0, idx_ji, x_kj)

        for layer in self.layers_before_skip:
            x_kj = layer(x_kj)

        x2 = x_kj + x_ji

        return x2


class DimeNet(nn.Module):
    def __init__(self, hidden_dim, num_bilinear, radius, act="ReLU",
                 num_spherical=6, num_radial=16, num_before_skip=1, num_after_skip=2):
        super(DimeNet, self).__init__()
        self.act = get_activation_func(act)
        self.atom_feature_dim = 133
        self.bond_feature_dim = 14
        self.hidden_dim = hidden_dim
        self.bias = False
        self.input_dim = 133 + 14


        self.num_MPNN_layer = radius

        self.dis_emb = BesselBasisLayer(num_radial=num_radial, cutoff=8.0)
        self.angle_emb = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=8.0)

        self.W_i_distance = EmbeddingBlock(num_radial=num_radial, hidden_channels=hidden_dim, act=act)
        self.W_i1 = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.W_i2 = nn.Linear(2 * self.hidden_dim, self.hidden_dim, bias=True)

        self.W_o = nn.Linear(self.atom_feature_dim + self.hidden_dim, self.hidden_dim)

        self.Interaction_layer = torch.nn.ModuleList([
            InteractionBlock(hidden_channels=hidden_dim, num_bilinear=num_bilinear,
                             num_spherical=num_spherical, num_radial=num_radial,
                             num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                             act=act)
            for _ in range(self.num_MPNN_layer)])

    def forward(self, G, gm):
        dist, angle, torsion, i, j, idx_kj, idx_ji, \
        incomebond_edge_ids, incomebond_index_to_atom = gm

        # G.edata["edge_feature"] = G.edata["edge_feature"].float()

        # def initial_func(edges):  # fea(j)+fea(j-->i) num_bonds x (133+14)
        #     return {'initial_feature':
        #             torch.cat((edges.src["atom_feature"], edges.data["edge_feature"]), dim=1)}

        # G.apply_edges(initial_func)

        initial_bonds = torch.cat((G.ndata["atom_feature"][j],G.edata["edge_feature"]),dim = 1) # fea(j)+fea(j-->i) num_bonds x (133+14)

        inputs = self.W_i1(initial_bonds)  # num_bonds x hidden_size
        message = self.act(inputs)  # num_bonds x hidden_size

        # 获得基于化学先验的初始表征
        # feats_che = G.edata["initial_feature"]
        # feats_che = self.act(self.W_i1(feats_che))  # num_bonds x hidden_size


        # 基于距离先验的初始表征
        atom_type = G.ndata["atom_feature"][:, 0:100].nonzero()[:, 1]
        rbf = self.dis_emb(dist)
        sbf = self.angle_emb(dist, angle, idx_kj)
        rbf = self.W_i_distance(atom_type, rbf, i, j)  # num_bonds x hidden_size
        # feats = torch.cat((feats_che, feats_dis), dim=1)
        # feats = self.act(self.W_i2(feats))  # 每一条边初始化的表征 num_bonds x hidden_size


        for layer in self.Interaction_layer:
            message = layer(message, rbf, sbf, idx_kj, idx_kj)


        num_atoms = G.num_nodes()
        atom_message = torch.zeros(num_atoms, self.hidden_dim).to(G.device)
        incomebond_hidden = message[incomebond_edge_ids]
        atom_message = atom_message.index_add_(0, incomebond_index_to_atom, incomebond_hidden)

        a_input = torch.cat([G.ndata["atom_feature"], atom_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act(self.W_o(a_input))  # num_atoms x hidden
        # atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens
