#!bin/bash

env_name=tinaface7

if [[ -d ~/miniconda3 ]]
then
    conda_source=~/miniconda3/etc/profile.d/conda.sh
elif [[ -d ~/anaconda3 ]]
then
    conda_source=~/anaconda3/etc/profile.d/conda.sh
fi

conda create -n ${env_name} python=3.7 -y

source ${conda_source}
conda activate ${env_name}

#conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
pip install -r requirements/build.txt
python -m pip install cython numpy opencv-python pyopengl
pip install -v -e .
