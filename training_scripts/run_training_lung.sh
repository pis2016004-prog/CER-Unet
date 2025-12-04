#!/bin/sh
cd /drive/Brats2017/ICE-Net

DATASET_PATH=../DATASET/DATASET_Lungs/DATASET_Lungs

export PYTHONPATH=./
export RESULTS_FOLDER=./ISBI_CAP_LUNG
export unetr_pp_preprocessed=../DATASET/DATASET_Lungs/DATASET_Lungs/unetr_pp_raw/unetr_pp_raw_data/Task06_Lung
export unetr_pp_raw_data_base=../DATASET/DATASET_Lungs/DATASET_Lungs/unetr_pp_raw

python icenet/run/run_training.py 3d_fullres icenet_trainer_lung 6 0 --continue_training
