import dgl
import dgl.nn.pytorch as dglnn
import pickle
import torch
from math import sqrt, pi as PI

def pkl_load(file_name):
    """
    load a pickle file

    :param file_name: The path of the pickle file of a ligand-protein complex.

    :return graph: A dgl graph of the ligand-protein complex.
            embedding: the ligand-protein interaction embedding computed by schrodinger.
            atomnum_of_ligand: the number of atoms of the ligand.
    """
    pickle_file = open(file_name,'rb')
    graph,embedding,atomnum_of_ligand = pickle.load(pickle_file)
    return graph,embedding,atomnum_of_ligand


def get_nearst_node(pos, sges):
    """
    find the nearest atoms of each atom.

    :param pos: a 2-D tensor that saves the coordinates of atoms in a graph.
           segs: if there are more than one graph, segs is a list that saves number of atoms of each graph.

    :return adj_nearest: the nearest atoms (col) of each atom (row). 
            adj_nearest2: the second near atom (col) of each atom (row). 
    """
    ref_indentity = dgl.segmented_knn_graph(pos,1,sges).adj(transpose=True) 
    # 最近的原子是原子本身，。。。。
    adj_nearest = dgl.segmented_knn_graph(pos,2,sges).adj(transpose=True)  
    # 转置之后，行为0-n的原子，列表示和该原子力的最近的原子
    adj_nearest2 = dgl.segmented_knn_graph(pos,3,sges).adj(transpose=True)

    adj_nearest2 = adj_nearest2.to_dense() - adj_nearest.to_dense()
    adj_nearest2 = adj_nearest2.to_sparse()
    
    adj_nearest = adj_nearest.to_dense() - ref_indentity.to_dense()
    adj_nearest = adj_nearest.to_sparse()
    
    return adj_nearest, adj_nearest2


def xyztodata(g):
	"""
	convert the rectangular coordinate system (x, y, z) to polar coordinates system (d, angle, torsion)
	extract the corresponding relations between k-->j and j-->i

	:param file_name: The path of the pickle file of a ligand-protein complex.

	:return dist: the distance (or lenth) of each j-->i [808]. 
			angle: the polar angle between j-->i and k-->j [1098].
			torsion: the direction angle between j-->i and k-->j [1098].
			i: the destination atom index of each bond [808].
			j: the source atom index of each bond [808].
			idx_kj: the bond index of each k-->j [1098].
			idx_ji: the bond index of each j-->i [1098] which is corresponding with the idx_kj.
	"""
	# graphs = []
	# for num in range(len(file_names)):
	# 	_g = pkl_load(file_names[num])[0]
	# 	graphs.append(_g)
	# g = dgl.batch(graphs)

	atom = g.nodes()
	incomebond_edge_ids = g.edge_ids(g.in_edges(atom)[0], g.in_edges(atom)[1])
	l = torch.tensor(range(g.num_nodes())).to(g.device)
	incomebond_index_to_atom = l.repeat_interleave(g.in_degrees(atom)) 

	j,i = g.edges()  # j --> i
	pos = g.ndata['atom_coordinate'] # coordinates of atoms in the dgl graph

	dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt() # the lenth of each ji

	number_of_k = []
	k_include_i = []
	for each_j in j:
		k_of_j = g.predecessors(each_j) # each k of j,即计算每个j原子的起始原子k(此时包括i)
		number_of_k.append(len(k_of_j))
		k_include_i.extend(list(k_of_j))

	idx_i = i.repeat_interleave(torch.tensor(number_of_k).to(g.device))  
	idx_j = j.repeat_interleave(torch.tensor(number_of_k).to(g.device))
	idx_k = torch.tensor(k_include_i).to(g.device)

	mask = idx_i != idx_k  # 除去 i-->j这条边
	idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

	# Edge index (k-->j, j-->i) for triplets.
	idx_ji  = g.edge_ids(idx_j,idx_i)
	idx_kj = g.edge_ids(idx_k,idx_j)

	# 计算极角 Calculate angles.
	pos_ji = pos[idx_i] - pos[idx_j]
	pos_jk = pos[idx_k] - pos[idx_j]
	a = (pos_ji * pos_jk).sum(dim=-1) # cos_angle * |pos_ji| * |pos_jk|
	b = torch.cross(pos_ji, pos_jk).norm(dim=-1) # |sin_angle| * |pos_ji| * |pos_jk|
	angle = torch.atan2(b, a)  # 角度

	# 计算方位角 Calculate torsions.
	adj_nearest, adj_nearest2 = get_nearst_node(pos, list(g.batch_num_nodes()))

	adj_nearest_row = adj_nearest.to_dense()[idx_j].to_sparse()    # 这样获得的矩阵，每一行表示距离j原子最近的原子
	adj_nearest2_row = adj_nearest2.to_dense()[idx_j].to_sparse()
	idx_k_n = adj_nearest_row.to_dense().nonzero()[:,1].to(g.device)    # 距离j原子最近的原子的索引
	idx_k_n2 = adj_nearest2_row.to_dense().nonzero()[:,1].to(g.device)  # 距离j原子第二近的原子的索引
	mask = idx_k_n == idx_i       
	idx_k_n[mask] = idx_k_n2[mask]   # 如果j最近的原子是i，那么提取第二近的原子（不一定是k中的一个）

	pos_jk = pos[idx_k] - pos[idx_j]       # 目标k原子的距离
	pos_ji = pos[idx_i] - pos[idx_j]       # ji的距离
	pos_j0 = pos[idx_k_n] - pos[idx_j]     # j和最近的k的距离
	dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
	plane1 = torch.cross(pos_ji, pos_j0)
	plane2 = torch.cross(pos_ji, pos_jk)
	a = (plane1 * plane2).sum(dim=-1) # cos_angle * |plane1| * |plane2|
	b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
	torsion = torch.atan2(b, a) # -pi to pi
	torsion[torsion<=0]+=2*PI # 0 to 2pi

	return dist, angle, torsion, i, j, idx_kj, idx_ji, incomebond_edge_ids, incomebond_index_to_atom 