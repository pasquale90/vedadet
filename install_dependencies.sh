#!bin/bash

env_name=tinaface

conda create -n ${env_name} python=3.7 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${env_name}
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements/build.txt
pip install -v -e .
