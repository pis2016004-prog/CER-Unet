#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_tumor
export unetr_pp_preprocessed="$DATASET_PATH"/unetr_pp_raw/unetr_pp_raw_data/Task03_tumor
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python cernet/run/run_training.py 3d_fullres cernet_trainer_tumor 3 0
