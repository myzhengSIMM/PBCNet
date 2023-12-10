#!/bin/bash

device=3
fep="fep1"

encoder_type="Bind"
batch_size=30
loss_function="mse"
continue_learning=0
hidden_dim=96
p_dropout=0.1

result_file="_finetune_fep2_ref6_100_20221105"

train_path="/home/yujie/leadopt/data/train_files_uni/mean_600000train_all_pair_withoutFEP.csv"
val_path="/home/yujie/leadopt/data/train_files_uni/train_1_pair.csv"
fold="mse_lr5-7_p0.1"
log_frequnency=100
init_lr=0.0000005
max_lr=0.0000005
final_lr=0.0000005

seed=222
degree_information=0
radius=3
readout_type="AttFP"

cs=0
GCN=1
two_task=1


retrain=1
retrain_model_path="/code/PBCNet.pth"


# rcs_sub3_angle_and_dist*0.0005_3



python ./finetune.py \
--device ${device} \
--encoder_type ${encoder_type} \
--batch_size ${batch_size} \
--loss_function ${loss_function} \
--continue_learning ${continue_learning} \
--hidden_dim  ${hidden_dim} \
--train_path ${train_path} \
--val_path ${val_path} \
--p_dropout ${p_dropout} \
--result_file ${result_file} \
--log_frequency ${log_frequnency} \
--max_lr ${max_lr} \
--init_lr ${init_lr} \
--final_lr ${final_lr} \
--retrain ${retrain} \
--retrain_model_path ${retrain_model_path} \
--degree_information ${degree_information} \
--cs ${cs} \
--fold ${fold} \
--seed ${seed} \
--radius ${radius} \
--readout_type ${readout_type} \
--GCN_ ${GCN} \
--two_task ${two_task} \
--which_fep ${fep}