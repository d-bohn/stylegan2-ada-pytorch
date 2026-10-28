import os
import pathlib

BASE_FOLDER = str(pathlib.Path(__file__).parent.parent.resolve())
# BASE_FOLDER = '/mnt/storage/Documents/project_envs/stylegan-ada-pytorch'

#required: definitely edit these!
dataset_path = '/mnt/data1/images/Neutral_Faces/namfhq/namfhq_processed_less_less_blur'
#resume_from =  '/mnt/data1/pretrained_models/stylegan2/metfaces.pkl' # or try ffhq, wikiart
resume_from =  '/mnt/data1/pretrained_models/stylegan2-training/namfhq-v2/namfhq-v2-gamma10/network-snapshot-000496.pkl'
aug_strength = 0.369
train_count = 496
mirror_x = True
#mirror_y = False

#optional: you might not need to edit these
gamma_value = 10.0 # 50?
augs = 'bg'
config = '24gb-gpu-complex' # '24gb-2gpu-complex'
snapshot_count = 4
nkimg = 10000
metrics = 'fid50k_full'

# exec(open(f'{BASE_FOLDER}/train.py').read())
os.system(f'python {BASE_FOLDER}/train.py \
          --gpus=1 \
          --cfg={config} \
          --metrics=None \
          --outdir=/mnt/data1/pretrained_models/stylegan2-training/namfhq-v2 \
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
