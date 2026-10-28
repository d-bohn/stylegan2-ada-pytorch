#!/bin/bash

# Run the gen_images.py script using Docker:
docker run --gpus 1 -it --rm --user $(id -u):$(id -g) \
    --shm-size=8g \
    -v `pwd`:/scratch  \
    -v /mnt/data1:/mnt/data1 \
    -v /mnt/storage:/mnt/storage \
    -v /home/dan:/home/dan \
    --workdir /scratch \
    -e HOME=/scratch \
    -e TORCH_EXTENSIONS_DIR=/home/dan/.cache/torch_extensions \
    stylegan2:latest \
    /bin/bash