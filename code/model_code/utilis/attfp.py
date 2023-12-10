# import package
import dgl
from dgl.data import DGLDataset
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

import os
import sys
sys.path.append("/home/yujie/AIcode/utilis/")

from trick import Writer
from scalar import StandardScaler
from initial import initialize_weights
from function import get_loss_func

cpu_num = 1
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
        '--loss_func', type=str, default="smoothl1", help='The loss function used to train the model: mse, smoothl1, mve'
        )
    parser.add_argument(
        "--device", type=int, default=0, help="The number of device: 0,1,2,3 [on v100-2]"
        )
    # parser.add_argument('--del_minor_water',type=bool,default=True,help='if delete water with low (3) H-bonds numbers'
    #     )
    # parser.add_argument(
    #     '--delwater_hbond_cutoff',type=int,default=3,help='If specified, delete waters that do not make at least,this number H-bonds to non-waters. '
    #     )

    return parser

def pkl_load(file_name):
    pickle_file = open(file_name,'rb')
    graph,embedding,atomnum_of_ligand = pickle.load(pickle_file)
    return graph,embedding,atomnum_of_ligand

def Extend(list1,list2):
    list1.extend(list2)
    return list1

# define dataloader and collec function
def collate_fn(samples):

    ligand1_dir = [s.Ligand1.values[0] for s in samples]
    ligand2_dir = [s.Ligand2.values[0] for s in samples]
    graph1_list = [pkl_load(s)[0] for s in ligand1_dir]
    graph2_list = [pkl_load(s)[0] for s in ligand2_dir]

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
           file_name

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
class Dif_Activity_Predictor(nn.Module):
    def __init__(self, hidden_dim, radius, T, p_dropout, ffn_num_layers):
        super(Dif_Activity_Predictor, self).__init__()
        self.attfp = AttentiveFPGNN(133, 14, radius, hidden_dim, dropout=p_dropout)
        self.readout = AttentiveFPReadout(feat_size=hidden_dim,
                                          num_timesteps=T,
                                          dropout=p_dropout)

        ffn = [nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU()]
        for _ in range(ffn_num_layers-1):
            ffn.append(nn.Linear(hidden_dim, hidden_dim))
            ffn.append(nn.ReLU())
        ffn.append(nn.Linear(hidden_dim, 1))

        self.encoder =nn.Sequential(*ffn)

    def forward(self, g1, h1, e1, alist1,num1, g2, h2, e2,alist2, num2):
        h1 = F.relu(self.attfp(g1, h1, e1))
        h2 = F.relu(self.attfp(g2, h2, e2))

        with g1.local_scope():
            with g2.local_scope():
                g1.ndata['h'] = h1
                g2.ndata['h'] = h2

                ug1 = dgl.unbatch(g1)
                ug2 = dgl.unbatch(g2) 

                sg1 = dgl.batch([ dgl.node_subgraph(ug1[i], alist1[i][0:num1[i]]) for i in range(len(ug1)) ])
                sg2 = dgl.batch([ dgl.node_subgraph(ug2[i], alist2[i][0:num2[i]]) for i in range(len(ug2)) ])

                # dgl.node_subgraph(g1, num1 )
                # sg2 = dgl.node_subgraph(g2, num2 )

                hsg1 = self.readout(sg1, sg1.ndata['h'], False)
                hsg2 = self.readout(sg2, sg2.ndata['h'], False)
                zk = self.encoder(torch.cat([hsg1, hsg2, hsg1-hsg2], dim=-1))

                return zk

# search function
# SPACE = {
#     'hidden_dim': hp.quniform('hidden_dim', low=300, high=600, q=100),
#     'radius': hp.quniform('radius', low=2, high=6, q=1),
#     'T': hp.quniform('T', low=1, high=5, q=1),
#     'p_dropout': hp.quniform('dropout', low=0.0, high=0.5, q=0.05),
#     'init_lr': hp.loguniform('init_lr', low=np.log(1e-4), high=np.log(1e-2)),
#     'ffn_num_layers': hp.quniform('ffn_num_layers', low=2, high=4, q=1)
# }
# INT_KEYS = ['hidden_dim', 'radius', 'T', 'ffn_num_layers']
# n = 0
# DATADIR = os.path.join(os.environ["HOME"], "data", "five_fold_data")
# logger = Writer(os.path.join(SAVEDIR, "history.log"))

def func(device = 'cuda:0', label_scalar = None, loss_func = "smoothl1"):

    if label_scalar == None:
        SAVEDIR = os.path.join(os.environ["HOME"],"leadopt", "results", f"attfp_pair_{loss_func}_scalarFalse")
    else:
        SAVEDIR = os.path.join(os.environ["HOME"],"leadopt", "results", f"attfp_pair_{loss_func}_scalarTrue")

    logger_writer = Writer(os.path.join(SAVEDIR, "record.txt"))

    # for key in INT_KEYS:
    #     hyperparams[key] = int(hyperparams[key])

    # for k in hyperparams:
    #     logger_writer(f"{k}\t{hyperparams[k]}")
    #     logger_writer(" ")

    validation_roc_list  = []



    train_dataset = LeadOptDataset("/home/yujie/leadopt/data/ic50_graph_rmH/trapart_insplit_07.csv", label_scalar)

    label_scalar = train_dataset.label_scalar

    valid_dataset = LeadOptDataset("/home/yujie/leadopt/data/ic50_graph_rmH/valpart_insplit_07.csv")
    # test_dataset = CoCrystalDataset("/home/wangdingyan/data/five_fold_data/test.csv")

    train_dataloader = GraphDataLoader(train_dataset, collate_fn=collate_fn, batch_size=64, drop_last=False, shuffle=True)
    valid_dataloader = GraphDataLoader(valid_dataset, collate_fn=collate_fn, batch_size=64, drop_last=False, shuffle=True)
    # test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=128, drop_last=False,shuffle=False)
    # test_dataloader.smiles1, test_dataloader.smiles2 = test_dataset.smiles1_list, test_dataset.smiles2_list

    model = Dif_Activity_Predictor(300,
                                   4,
                                   3,
                                   0.9,
                                   3).to(device)
    initialize_weights(model)
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3, eps = 1e-4)
    loss_func = get_loss_func(loss_func)

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
            BATCH += 1
            logits = model(k[0].to(device),
                           k[0].ndata["atom_feature"].to(device),
                           k[0].edata["edge_feature"].to(device),
                           k[3].to(device),
                           k[5].to(device),
                           k[1].to(device),
                           k[1].ndata["atom_feature"].to(device),
                           k[1].edata["edge_feature"].to(device),
                           k[4].to(device),
                           k[6].to(device))

            loss   =  loss_func(logits.squeeze().float(), k[2].to(device).float())
            loss_for_ten_batch.append(loss.detach().cpu().item())
            loss_all_train.append(loss.detach().cpu().item())
            opt.zero_grad()
            loss.backward()
            opt.step()

            if len(loss_for_ten_batch) == 10:
                l = np.mean(loss_for_ten_batch)
                print(f"Epoch {EPOCH} Batch {BATCH}  Loss {l}")
                logger_writer(f"Epoch {EPOCH} Batch {BATCH}  Loss {l}")
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
                logits = model(k[0].to(device),
                               k[0].ndata["atom_feature"].to(device),
                               k[0].edata["edge_feature"].to(device),
                               k[3].to(device),
                               k[5].to(device),
                               k[1].to(device),
                               k[1].ndata["atom_feature"].to(device),
                               k[1].edata["edge_feature"].to(device),
                               k[4].to(device),
                               k[6].to(device))

                valid_prediction.extend(logits.tolist())
                valid_labels.extend(k[2].tolist())
                valid_1_labels.extend(k[9].tolist())
                ref_2_labels.extend(k[10].tolist())
                rank.extend(k[11].tolist())
                file.extend(k[12])

            if label_scalar == None:
                pass
            else:
                valid_prediction = label_scalar.inverse_transform(valid_prediction)


            mae = mean_absolute_error(valid_labels, valid_prediction)
            mse = mean_squared_error(valid_labels, valid_prediction)

            pre_abs_pic50 = np.array(valid_prediction).flatten() + np.array(ref_2_labels)

            abs_mse = mean_squared_error(valid_1_labels, list(pre_abs_pic50))

            file_to_p = defaultdict(list)
            for pre, lab, r, f in zip(pre_abs_pic50,valid_1_labels,rank , file):
                file_to_p[f].append([pre, lab, r])
            
            spearman = []
            pearson = []
            lenth = []
            files_ = []
            for f in file_to_p.keys():
                _df = pd.DataFrame(file_to_p[f], columns = ['a','b','c'])
                _df = _df.groupby('c')[['a','b']].mean().reset_index()

                if len(_df) >= 20:
                    files_.append(f)
                    lenth.append(len(_df))

                    spear = _df[["a","b"]].corr(method='spearman')
                    spearman.append(spear.iloc[0,1])

                    pear = _df[["a","b"]].corr(method='pearson')
                    pearson.append(pear.iloc[0,1])

            csv_save_dir = os.path.join(SAVEDIR, f"2.3_scalar_corr_{EPOCH}.csv")
            corr_df = pd.DataFrame({'file_name':files_,\
                                    'spearman':spearman,\
                                    'pearson':pearson,\
                                    'num_of_val_data':lenth})
            corr_df.to_csv(csv_save_dir, index = 0)

            logger_writer("  ")
            logger_writer(f"Epoch {EPOCH}")
            logger_writer(f"Validation Set mae {mae}")
            logger_writer(f"Validation Set mse {mse}")
            logger_writer(f"Train Set mse {np.nanmean(loss_all_train)}")
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
                logger_writer(f"Best Validation mse {stop_metric}")

            if not_change_epoch >= 10:
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
    loss_function = config.loss_func
    cuda = "cuda:" + str(config.device)


    if use_scalar == True:
        scalar = StandardScaler()
    else:
        scalar = None

    func(device=cuda, label_scalar = scalar, loss_func=loss_function)








