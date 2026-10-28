import os
import pathlib

BASE_FOLDER = str(pathlib.Path(__file__).parent.parent.resolve())
# BASE_FOLDER = '/mnt/storage/Documents/project_envs/stylegan-ada-pytorch'

#required: definitely edit these!
dataset_path = '/mnt/storage1/images/N-fwhq-1024.zip'
resume_from = 'ffhq1024'
aug_strength = 0.0
train_count = 0
mirror_x = True
#mirror_y = False

#optional: you might not need to edit these
gamma_value = 50.0
augs = 'bg'
config = '24gb-gpu'
snapshot_count = 4

# exec(open(f'{BASE_FOLDER}/train.py').read())
os.system(f'python {BASE_FOLDER}/train.py \
          --gpus=1 \
          --cfg={config} \
          --metrics=None \
          --outdir={BASE_FOLDER}/results \
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