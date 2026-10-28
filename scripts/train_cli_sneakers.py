import os
import pathlib

BASE_FOLDER = str(pathlib.Path(__file__).parent.parent.resolve())
# BASE_FOLDER = '/mnt/storage/Documents/project_envs/stylegan-ada-pytorch'

#required: definitely edit these!
dataset_path = '/mnt/storage1/images/sneakers/sneaker-image-dataset.zip'
resume_from =  '/mnt/storage1/pretrained_models/stylgan2-training/sneakers/00001-sneaker-image-dataset-mirror-24gb-gpu-gamma50-bg-resumecustom/network-snapshot-001120.pkl'
aug_strength = 0.652
train_count = 1120
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
          --outdir=/mnt/storage1/pretrained_models/stylgan2-training/sneakers/ \
          --data={dataset_path} \
          --snap={snapshot_count} \
          --resume={resume_from} \
          --augpipe={augs} \
          --initstrength={aug_strength} \
          --gamma={gamma_value} \
          --mirror={mirror_x} \
          --mirrory=False \
          --nkimg={train_count}'
          )
