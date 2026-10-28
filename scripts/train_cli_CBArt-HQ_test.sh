#!/usr/bin/bash

# export CUDA_VISIBLE_DEVICES=1

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch/'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/CBArt-HQ-v2/'

#required: definitely edit these!
dataset_path='/mnt/data1/images/art/for_training_sub'

# resume_from='/mnt/data1/pretrained_models/stylegan2-training/CBArt-HQ-v2/00000-for_training-mirror-24gb-gpu-complex-gamma75-kimg25000-bg/saved_models/network-snapshot-011920.pkl'
resume_from='/mnt/data1/pretrained_models/stylegan2/wikiart.pkl'
aug_strength=0.00
train_count=0
mirror_x=True

#optional: you might not need to edit these
gamma_value=75 # scales quadratically
augs='bg' # or bg, bgc, bgcfn
config='24gb-gpu-complex' # 'auto2'
snapshot_count=10
total_train=25000
metrics=None
num_workers=3

python $BASE_FOLDER/train.py \
          --gpus=1 \
          --seed=1999 \
          --cfg=$config \
          --metrics=$metrics \
          --outdir=$OUT_FOLDER \
          --data=$dataset_path \
          --snap=$snapshot_count \
          --augpipe=$augs \
          --gamma=$gamma_value \
          --mirror=$mirror_x \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers \
          --initstrength=$aug_strength \
          --resume=$resume_from #\
          # --batch=$batch_size \
          # --freezed=$freezed
