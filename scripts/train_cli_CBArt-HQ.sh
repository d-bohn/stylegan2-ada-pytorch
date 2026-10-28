#!/usr/bin/bash

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch/'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/CBArt-HQ-v2/'

#required: definitely edit these!
dataset_path='/mnt/data1/images/art/for_training_sub'

resume_from='/mnt/data1/pretrained_models/stylegan2-training/CBArt-HQ-v2/00003-stylegan2-for_training_sub-gpus1-batch32-gamma52.4288/network-snapshot-007440.pkl'
# resume_from='/mnt/data1/pretrained_models/stylegan2/wikiart.pkl'
aug_strength=0.259
train_count=7440
mirror_x=True

#optional: you might not need to edit these
gamma_value=52.4288 # scales quadratically
aug_type='ada' # 'noaug' or 'ada'; default is 'ada'
augs='bgc' # or bg, bgc, bgcfn
ada_target=0.4 # default is 0.6
config='24gb-gpu-complex' # '24gb-2gpu-complex' or 'auto2' see train.py for more configs
snapshot_count=20
total_train=25000
metrics=fid50k_full # or fic50k
num_workers=4
glearning=0.002
dlearning=0.002
batch_size=12
# freezed=10 # or 0, 13

python $BASE_FOLDER/train.py \
          --gpus=1 \
          --cfg=$config \
          --metrics=$metrics \
          --outdir=$OUT_FOLDER \
          --data=$dataset_path \
          --snap=$snapshot_count \
          --aug=$aug_type \
          --augpipe=$augs \
          --target=$ada_target \
          --gamma=$gamma_value \
		  --glrate=$glearning \
          --dlrate=$dlearning \
          --mirror=$mirror_x \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers \
          --initstrength=$aug_strength \
          --resume=$resume_from #\
          # --batch=$batch_size \
          # --freezed=$freezed
