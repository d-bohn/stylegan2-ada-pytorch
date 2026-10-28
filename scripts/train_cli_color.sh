#!/usr/bin/bash

BASE_FOLDER='/mnt/storage/Documents/project_envs/stylegan2-ada-pytorch'
OUT_FOLDER='/mnt/data1/pretrained_models/stylegan2-training/color-space/512px_no_gradients_training/'

#required: definitely edit these!
# dataset_path='/mnt/data1/images/color/all_colorspaces_no_gradients_combined_512px.zip'
dataset_path='/mnt/data1/images/color/color_images_no_gradients_512px/perceptual.zip'

# resume_from='/mnt/data1/pretrained_models/stylegan2-training/color-space/mercury-color_images_128px-002800-48gb-gpu-gamma0.1024-kimg25000-batch32-noaug.pkl'
# resume_from='/mnt/data1/pretrained_models/stylegan2-training/color-space/512px_no_gradients_training/00000-all_colorspaces_no_gradients_combined_512px-24gb-gpu-color-gamma1-kimg25000-batch16-ada-target0.4-bgcfn/network-snapshot-002600.pkl'
resume_from='/mnt/data1/pretrained_models/stylegan2-training/color-space/512px_no_gradients_training/00005-all_colorspaces_no_gradients_combined_512px-24gb-gpu-color-gamma1-kimg25000-batch16-ada-target0.4-bgcfn-resumecustom/network-snapshot-006000.pkl'
mirror_x=False
mirror_y=False

#optional: you might not need to edit these
gamma_value=1 # scales quadratically
aug_type='ada' # 'noaug' or 'ada'; default is 'ada'
augs='bgcfn' # or bg, bgc, bgcfn
ada_target=0.4 # default is 0.6
aug_strength=0.185 # default is 0.
config='24gb-gpu-color'
snapshot_count=50
total_train=25000
train_count=6000
metrics=fid50k_full
num_workers=4
batch_size=16 # 8, 16, 32, 64
# learning_rate=0.001
glearning=0.00001
dlearning=0.00001
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
		  --batch=$batch_size \
		  --glrate=$glearning \
          --dlrate=$dlearning \
          --mirror=$mirror_x \
          --mirrory=$mirror_y \
          --nkimg=$train_count \
          --kimg=$total_train \
          --workers=$num_workers \
          --initstrength=$aug_strength \
          --resume=$resume_from #\
          # --freezed=$freezed
