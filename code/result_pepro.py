import argparse
import os
import random
import time
from collections import defaultdict

code_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(code_path + '/model_code')

code_path = code_path.rsplit("/", 1)[0]

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Dataloader.dataloader import collate_fn, LeadOptDataset,LeadOptDataset_test
from train.predict import predict
from ReadoutModel.readout_bind import DMPNN
# from ReadoutModel.readout_bind_deltadelta import DMPNN
from utilis.function import get_loss_func
from utilis.initial import initialize_weights
from utilis.scalar import StandardScaler
from utilis.scheduler import NoamLR_shan
from utilis.trick import Writer
from utilis.utilis import gm_process




test_file_names_FEP2 = [i for i in os.listdir(f"/data/test_set_fep+_graph_rmH_I/") if i != "input_files"]
test_file_names_FEP1 = [i for i in os.listdir(f"/data/test_set_fep_graph_rmH_I/") if i != "input_files"]

test_rmse_loader_FEP2 = []
test_corr_loader_FEP2 = []
for y in test_file_names_FEP2:
    test_dataset = LeadOptDataset(f"/data/test_set_fep+_graph_rmH_I/input_files/0_reference/train_files/{y}.csv")
    test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=60,
                                        drop_last=False, shuffle=False,
                                         pin_memory=True)
    test_rmse_loader_FEP2.append(test_dataloader)

    test_dataset = LeadOptDataset(f"/data/test_set_fep+_graph_rmH_I/input_files/1_reference/train_files/{y}.csv")
    test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=60,
                                        drop_last=False, shuffle=False,
                                        pin_memory=True)
    test_corr_loader_FEP2.append(test_dataloader)


test_rmse_loader_FEP1 = []
test_corr_loader_FEP1 = []
for y in test_file_names_FEP1:
    test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep_graph_rmH_I/input_files/0_reference/train_files/{y}.csv")
    test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=60,
                                        drop_last=False, shuffle=False,
                                        pin_memory=True)
    test_rmse_loader_FEP1.append(test_dataloader)

    test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep_graph_rmH_I/input_files/1_reference/train_files/{y}.csv")
    test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=60,
                                        drop_last=False, shuffle=False,
                                        pin_memory=True)
    test_corr_loader_FEP1.append(test_dataloader)




device = "cpu"
model = torch.load("/code/PBCNet.pth",map_location="cpu")


    #  =================== FEP+ =================
# rmse
rmse_gs = []

file_name = []
for num_loader in range(len(test_rmse_loader_FEP2)):
    loader = test_rmse_loader_FEP2[num_loader]
    file_nm = test_file_names_FEP2[num_loader]
    file_name.append(file_nm)

    # if num_loader == 0:
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict(model, loader, device)



    rmse_gs.append(rmse_g)

    if num_loader == 0:
        prediction_of_FEP2 = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                            f"label_ic50_{file_nm}": valid_labels})
    else:
        prediction_of_FEP2_ = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                            f"label_ic50_{file_nm}": valid_labels})
        prediction_of_FEP2 = pd.merge(prediction_of_FEP2, prediction_of_FEP2_, how="outer",right_index=True,left_index=True)



# corr
spearmans = []
pearsons = []
kendalls = []

spearmans_var = []
pearsons_var = []
kendalls_var = []

spearmans_min = []
pearsons_min = []
kendalls_min = []

spearmans_max = []
pearsons_max = []
kendalls_max = []

for num_loader in range(len(test_corr_loader_FEP2)):
    loader = test_corr_loader_FEP2[num_loader]
    file_nm = test_file_names_FEP2[num_loader]
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, _ ,_,_= predict(model, loader, device)

    df = pd.read_csv(f"{code_path}/data/test_set_fep+_graph_rmH_I/input_files/1_reference/train_files/{file_nm}.csv")
    df["predict_dmpnnmve_pic50"] = valid_prediction

    abs_label = df.Lable1.values
    abs_predict = np.array(df.predict_dmpnnmve_pic50.values).astype(float) + np.array(df.Lable2.values).astype(
        float)

    df["abs_label_p"] = abs_label
    df["abs_predict_p"] = abs_predict

    # =================以PIC50为单位====================
    reference_num = df.reference_num.values
    ligand1_num = df.Ligand1_num.values
    _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_nm}": abs_label, f"abs_predict_{file_nm}": abs_predict, f"ligand1_num_{file_nm}": ligand1_num})

    # ================ 用来画散点图的 ==============
    if num_loader == 0:
        corr_of_FEP2 = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
    else:
        corr_of_FEP2_ = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
        corr_of_FEP2 = pd.merge(corr_of_FEP2, corr_of_FEP2_, how="outer",right_index=True,left_index=True)


    _df_group = _df.groupby('reference_num')  # [['abs_label', 'abs_predict']].mean().reset_index()

    spearman_ = []
    pearson_ = []
    kendall_ = []
    for _, _df_onegroup in _df_group:
        spearman = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='spearman').iloc[0, 1]
        pearson = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='pearson').iloc[0, 1]
        kendall = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='kendall').iloc[0, 1]
        spearman_.append(spearman)
        pearson_.append(pearson)
        kendall_.append(kendall)
    spearmans.append(np.mean(spearman_))
    pearsons.append(np.mean(pearson_))
    kendalls.append(np.mean(kendall_))

    spearmans_var.append(np.var(spearman_))
    pearsons_var.append(np.var(pearson_))
    kendalls_var.append(np.var(kendall_))

    spearmans_min.append(np.min(spearman_))
    pearsons_min.append(np.min(pearson_))
    kendalls_min.append(np.min(kendall_))

    spearmans_max.append(np.max(spearman_))
    pearsons_max.append(np.max(pearson_))
    kendalls_max.append(np.max(kendall_))

for m_ in range(len(test_file_names_FEP2)):
    file_nm = test_file_names_FEP2[m_]
    rmse__ = rmse_gs[m_]
    s_ = spearmans[m_]
    p_ = pearsons[m_]
    k_ = kendalls[m_]
    s_var_ = spearmans_var[m_]
    p_var_ = pearsons_var[m_]
    k_var_ = kendalls_var[m_]

    s_max_ = spearmans_max[m_]
    p_max_ = pearsons_max[m_]
    k_max_ = kendalls_max[m_]

    s_min_ = spearmans_min[m_]
    p_min_ = pearsons_min[m_]
    k_min_ = kendalls_min[m_]

    print(f"{file_nm},RMSE:{rmse__},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                    pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_},kendall:{k_},kendall_var:{k_var_},\
                    kendall_min:{k_min_},kendalls_max:{k_max_}")

print(f"FEP,RMSE:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},spearman_var:{np.mean(spearmans_var)},spearmans_min:{np.mean(spearmans_min)},spearmans_max:{np.mean(spearmans_max)},\
                    pearson:{np.mean(pearsons)}, pearsons_var:{np.mean(pearsons_var)},pearson_min:{np.mean(pearsons_min)},pearsons_max:{np.mean(pearsons_max)},kendall:{np.mean(kendalls)}, \
                    kendall_var:{np.mean(kendalls_var)},kendall_min:{np.mean(kendalls_min)},kendalls_max:{np.mean(kendalls_max)}")
fep1_spearmans = np.mean(spearmans)

#  =================== FEP =================
# rmse
rmse_gs = []

file_name = []
for num_loader in range(len(test_rmse_loader_FEP1)):
    loader = test_rmse_loader_FEP1[num_loader]
    file_nm = test_file_names_FEP1[num_loader]
    file_name.append(file_nm)
    # if num_loader == 0:
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict( model, loader,
                                                                                            device)


    rmse_gs.append(rmse_g)
    if num_loader == 0:
        prediction_of_FEP1 = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                            f"label_ic50_{file_nm}": valid_labels})
    else:
        prediction_of_FEP1_ = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                            f"label_ic50_{file_nm}": valid_labels})
        prediction_of_FEP1 = pd.merge(prediction_of_FEP1, prediction_of_FEP1_, how="outer",right_index=True,left_index=True)


prediction_of_FEP = pd.merge(prediction_of_FEP1, prediction_of_FEP2, how="outer",right_index=True,left_index=True)



# corr
spearmans = []
pearsons = []
kendalls = []

spearmans_var = []
pearsons_var = []
kendalls_var = []

spearmans_min = []
pearsons_min = []
kendalls_min = []

spearmans_max = []
pearsons_max = []
kendalls_max = []
for num_loader in range(len(test_corr_loader_FEP1)):
    loader = test_corr_loader_FEP1[num_loader]
    file_nm = test_file_names_FEP1[num_loader]
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, _ ,_,_= predict(model, loader, device)

    df = pd.read_csv(
        f"{code_path}/data/test_set_fep_graph_rmH_I/input_files/1_reference/train_files/{file_nm}.csv")
    df["predict_dmpnnmve_pic50"] = valid_prediction

    abs_label = df.Lable1.values
    abs_predict = np.array(df.predict_dmpnnmve_pic50.values).astype(float) + np.array(df.Lable2.values).astype(
        float)

    df["abs_label_p"] = abs_label
    df["abs_predict_p"] = abs_predict

    # ================= PIC50 unit====================
    reference_num = df.reference_num.values
    ligand1_num = df.Ligand1_num.values
    _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_nm}": abs_label, f"abs_predict_{file_nm}": abs_predict, f"ligand1_num_{file_nm}": ligand1_num})

    # ================ for rmse =================
    if num_loader == 0:
        corr_of_FEP1 = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
    else:
        corr_of_FEP1_ = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
        corr_of_FEP1 = pd.merge(corr_of_FEP1, corr_of_FEP1_, how="outer",right_index=True,left_index=True)

    # ================ the computation of ranking related indicators ==============
    _df_group = _df.groupby('reference_num')  # [['abs_label', 'abs_predict']].mean().reset_index()

    spearman_ = []
    pearson_ = []
    kendall_ = []
    for _, _df_onegroup in _df_group:
        spearman = \
        _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='spearman').iloc[0, 1]
        pearson = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='pearson').iloc[
            0, 1]
        kendall = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='kendall').iloc[
            0, 1]
        spearman_.append(spearman)
        pearson_.append(pearson)
        kendall_.append(kendall)
    spearmans.append(np.mean(spearman_))
    pearsons.append(np.mean(pearson_))
    kendalls.append(np.mean(kendall_))

    spearmans_var.append(np.var(spearman_))
    pearsons_var.append(np.var(pearson_))
    kendalls_var.append(np.var(kendall_))

    spearmans_min.append(np.min(spearman_))
    pearsons_min.append(np.min(pearson_))
    kendalls_min.append(np.min(kendall_))

    spearmans_max.append(np.max(spearman_))
    pearsons_max.append(np.max(pearson_))
    kendalls_max.append(np.max(kendall_))

for m_ in range(len(test_file_names_FEP1)):
    file_nm = test_file_names_FEP1[m_]
    rmse__ = rmse_gs[m_]
    s_ = spearmans[m_]
    p_ = pearsons[m_]
    k_ = kendalls[m_]

    s_var_ = spearmans_var[m_]
    p_var_ = pearsons_var[m_]
    k_var_ = kendalls_var[m_]

    s_max_ = spearmans_max[m_]
    p_max_ = pearsons_max[m_]
    k_max_ = kendalls_max[m_]

    s_min_ = spearmans_min[m_]
    p_min_ = pearsons_min[m_]
    k_min_ = kendalls_min[m_]

    print(f"{file_nm},RMSE:{rmse__},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                    pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_},kendall:{k_},kendall_var:{k_var_},\
                    kendall_min:{k_min_},kendalls_max:{k_max_}")

print(f"FEP,RMSE:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},spearman_var:{np.mean(spearmans_var)},spearmans_min:{np.mean(spearmans_min)},spearmans_max:{np.mean(spearmans_max)},\
                    pearson:{np.mean(pearsons)}, pearsons_var:{np.mean(pearsons_var)},pearson_min:{np.mean(pearsons_min)},pearsons_max:{np.mean(pearsons_max)},kendall:{np.mean(kendalls)}, \
                    kendall_var:{np.mean(kendalls_var)},kendall_min:{np.mean(kendalls_min)},kendalls_max:{np.mean(kendalls_max)}")