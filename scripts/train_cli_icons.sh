#!/usr/bin/bash

export TORCH_CUDA_ARCH_LIST="7.5"
export GPU_VISIBLE_DEVICES=0

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/icons/icons_v3/'

#required: definitely edit these!
dataset_path='/mnt/data1/images/icons/img_collection/original_inverted'
# resume_from='/mnt/data1/pretrained_models/stylegan2-training/icons/icons_v3/00000-original_inverted-cond-24gb-gpu-complex-gamma12-kimg25000-batch16-noaug/network-snapshot-001200.pkl'
mirror_x=False
mirror_y=False
conditional=1

#optional: you might not need to edit these
gamma_value=3.28 # scales quadratically
aug_type='noaug' # 'noaug' or 'ada'; default is 'ada'
# augs='bgcfn' # or bg, bgc, bgcfn
# ada_target=0.6 # default is 0.6
# aug_strength=0 # default is 0.
config='24gb-gpu-complex'
snapshot_count=50
total_train=25000
train_count=1200
metrics=fid50k_full
num_workers=4
batch_size=16 # 8, 16, 32, 64
glearning=0.0018
dlearning=0.0024
# freezed=10 # or 0, 13

python $BASE_FOLDER/train.py \
          --gpus=1 \
          --cfg=$config \
          --metrics=$metrics \
          --outdir=$OUT_FOLDER \
          --data=$dataset_path \
          --cond=$conditional \
          --snap=$snapshot_count \
          --aug=$aug_type \
          --gamma=$gamma_value \
          --batch=$batch_size \
	  --glrate=$glearning \
          --dlrate=$dlearning \
          --mirror=$mirror_x \
          --mirrory=$mirror_y \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers # \
        #   --resume=$resume_from \


# Unused options:
        #   --augpipe=$augs \
        #   --target=$ada_target \
        #   --freezed=$freezed \
        #   --initstrength=$aug_strength
