#!/bin/bash

#SBATCH --job-name=coruja_train
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out


img_path=~/imgs/pi-coruja-model_latest.sif
singularity exec --nv $img_path bash -c "
cd ../src && python3 treinar_cnn.py --data-dir ../data/dataset_train --early-stop-patience 50
"
