#!/bin/sh
cd /drive/Brats2017/ICE-Net
DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=./
export RESULTS_FOLDER=./output_synapse_INCE2
export unetr_pp_preprocessed=../DATASET/DATASET_Synapse/unetr_pp_raw/unetr_pp_raw_data/Task02_Synapse
export unetr_pp_raw_data_base="$DATASET_PATH"/unetr_pp_raw

python icenet/run/run_training.py 3d_fullres icenet_trainer_synapse 2 0 
