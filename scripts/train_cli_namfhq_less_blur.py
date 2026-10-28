import os
import pathlib

BASE_FOLDER = str(pathlib.Path(__file__).parent.parent.resolve())
# BASE_FOLDER = '/mnt/storage/Documents/project_envs/stylegan-ada-pytorch'

#required: definitely edit these!
dataset_path = '/mnt/data1/images/Neutral_Faces/namfhq/namfhq_processed_less_blur/'
# resume_from = 'ffhq1024'
resume_from = '/mnt/data1/pretrained_models/stylgan2-training/namfhq_less_blur/2023-07-04_namfhq_less_blur/00001--mirror-24gb-2gpu-complex-gamma50-kimg12000-bg-resumeffhq1024/network-snapshot-000208.pkl'
aug_strength = 0.063
train_count = 208
mirror_x = True
#mirror_y = False

#optional: you might not need to edit these
gamma_value = 50.0
augs = 'bg'
config = '24gb-2gpu-complex'
snapshot_count = 4

# exec(open(f'{BASE_FOLDER}/train.py').read())
os.system(f'python {BASE_FOLDER}/train.py \
          --gpus=2 \
          --cfg={config} \
          --metrics=None \
          --outdir=/mnt/data1/pretrained_models/stylgan2-training/namfhq_less_blur/2023-07-04_namfhq_less_blur \
          --data={dataset_path} \
          --snap={snapshot_count} \
          --resume={resume_from} \
          --augpipe={augs} \
          --initstrength={aug_strength} \
          --gamma={gamma_value} \
          --mirror={mirror_x} \
          --mirrory=False \
          --kimg=12000 \
          --nkimg={train_count}'
          )
