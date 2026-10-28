#!/usr/bin/bash

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch/'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/sneakers-v2/'

#required: definitely edit these!
#dataset_path='/mnt/data1/images/Neutral_Faces/namfhq/namfhq_processed_less_less_blur'
dataset_path='/mnt/data1/images/sneakers/for_training'

resume_from='/mnt/data1/pretrained_models/stylegan2-training/sneakers/00002-sneaker-image-dataset-mirror-24gb-2gpu-complex-gamma50-bg-resumecustom/network-snapshot-001136.pkl' # or try ffhq1024
aug_strength=0.652
train_count=1136
mirror_x=False

#optional: you might not need to edit these
gamma_value=50 # scales quadratically
augs='bg' # or bg, bgc, bgcfn
config='24gb-gpu-complex' # '24gb-2gpu-complex'
snapshot_count=10
total_train=25000
metrics=fid50k_full
num_workers=3
# freezed=10 # or 0, 13

python $BASE_FOLDER/train.py \
          --gpus=1 \
          --cfg=$config \
          --metrics=$metrics \
          --outdir=$OUT_FOLDER \
          --data=$dataset_path \
          --snap=$snapshot_count \
          --resume=$resume_from \
          --augpipe=$augs \
          --initstrength=$aug_strength \
          --gamma=$gamma_value \
          --mirror=$mirror_x \
          --mirrory=False \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers # \
          # --freezed=$freezed
