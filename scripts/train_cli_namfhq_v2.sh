#!/usr/bin/bash

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch/'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/namfhq-v2/'

#required: definitely edit these!
dataset_path='/mnt/data1/images/Neutral_Faces/namfhq/namfhq_processed_less_less_blur'
resume_from='/mnt/data1/pretrained_models/stylegan2/stylegan2-ffhq-config-f.pkl'
#resume_from='/mnt/data1/pretrained_models/stylegan2-training/namfhq-v2/00000-namfhq_processed_less_less_blur-mirror-24gb-gpu-complex-gamma52-kimg1000-bgcfn-resumecustom/network-snapshot-001000.pkl'
#resume_from='ffhq1024' # or try wikiart
aug_strength=0.0
train_count=0
mirror_x=True

#optional: you might not need to edit these
gamma_value=75 # scales quadratically
augs='bg' # or bg, bgc, bgcfn
config='24gb-gpu-complex' # '24gb-2gpu-complex'
snapshot_count=10
total_train=25000
metrics=fid50k_full
num_workers=4
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
