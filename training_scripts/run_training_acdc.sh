#!/bin/sh
cd /drive/Brats2017/CER-UNet
DATASET_PATH=../DATASET/DATASET_Acdc

export PYTHONPATH=./
export RESULTS_FOLDER=./acdc_ISBICAP
export unetr_pp_preprocessed=../DATASET/DATASET_Acdc/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
#"$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC
export unetr_pp_raw_data_base=../DATASET/DATASET_Acdc/unetr_pp_raw

python cernet/run/run_training.py 3d_fullres icenet_trainer_acdc 1 0 

