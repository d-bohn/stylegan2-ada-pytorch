#!/usr/bin/bash

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch/'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/fontmania/'

#required: definitely edit these!
dataset_path='/mnt/data1/images/fonts_for_training_inv/'
# dataset_path='/mnt/data1/images/font_test/'

# resume_from='/mnt/data1/pretrained_models/stylegan2/stylegan2-ffhq-config-f.pkl'
resume_from='/mnt/data1/pretrained_models/stylegan2-training/fontmania/00004--cond-auto21-gamma8-kimg25000-batch32-noaug-resumecustom/network-snapshot-016346.pkl'
aug_strength=0.0
train_count=16346
mirror_x=False
conditional=1

#optional: you might not need to edit these
gamma_value=0.8192 # scales quadratically
augs=noaug # or ada, fixed
augopipe='bg' # or bg, bgc, bgcfn
config='auto2' # '24gb-2gpu-complex'
snapshot_count=20
total_train=25000
metrics=fid50k # or fid50k, fid50k_full, pr50k3_full, pr50k_full
num_workers=3
batch_size=32
# freezed=10 # or 0, 13

python $BASE_FOLDER/train.py \
          --gpus=1 \
          --cfg=$config \
          --cond=$conditional \
          --metrics=$metrics \
          --outdir=$OUT_FOLDER \
          --data=$dataset_path \
          --snap=$snapshot_count \
          --aug=$augs \
          --initstrength=$aug_strength \
          --gamma=$gamma_value \
          --mirror=$mirror_x \
          --mirrory=False \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers \
          --batch=$batch_size \
          --resume=$resume_from
          # --augpipe=$augpipe \
          # --freezed=$freezed
