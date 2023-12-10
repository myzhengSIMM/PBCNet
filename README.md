# PBCNet 

## 1 Environment
###### 1.1 python=3.7
###### 1.2 dgl； conda install -c dglteam dgl-cuda11.3=0.8.1
###### 1.3 biopython; conda install biopython
###### 1.4 rdkit； conda install -c conda-forge rdkit=2018.09.3
###### 1.5 openbabel；conda install openbabel -c conda-forge
###### 1.6 tqdm； conda install tqdm
###### 1.7 pytorch；conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -c pytorch
###### 1.8 conda install requests
###### 1.9 conda install scikit-learn
###### 1.10 conda install openpyxl
###### 1.11 pip install dgllife==0.2.6
###### 1.12 conda install pyg -c pyg 

## 2 How to finetune the PBCNet? 
set 'code/run_finetune.sh' as run file

## 3 code/model_code 
### The skeleton code of the PBCNet model.
###### AU-MPNN: code/model_code/Final/final.py
###### The whole PBCNet: code/model_code/ReadoutModel/readout_bind.py

## 4 results_in_our_article 
Summary of the outcome data reported in the article.

## 5 code/PBCNet.pth 
The trained PBCNet

## 6 data 
Note: The nature of pairwise input required for PBCNet results in one sample appearing in multiple sample pairs. Therefore, to reduce the time spent on data processing during training and prediction, we store most of the data as pickle files.
#### 6.1 FEP1 
The ligands in the FEP1 set on mol2 and sdf formats; the protein and pocket files on mol2 and pdb formats; and the computing results of intermolecular interactions.
#### 6.2 FEP2 
The ligands in the FEP2 set on mol2 and sdf formats; the protein and pocket files on mol2 and pdb formats; and the computing results of intermolecular interactions.
#### 6.3 test_set_fep_graph_rmH_I 
The corresponding pickle files of the FEP1 set.
#### 6.4 test_set_fep+_graph_rmH_I 
The corresponding pickle files of the FEP2 set.
#### 6.5 finetune_input_files 
The model input files (csv files) for finetune operation.
#### 6.6 ic50_graph_rmH_new 
The corresponding pickle files of the Training set.
#### 6.7 mean_600000train_all_pair_withoutFEP.csv 
The  model input files (csv files) for tarining.
#### 6.8 selection 
The ligands in the selection test set on mol2 and sdf formats and the protein and pocket files on mol2 and pdb formats.
#### 6.9 selection_graph 
The corresponding pickle files of the selection test set.
