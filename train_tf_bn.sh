#!/bin/bash
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate tinaface2

CUDA_VISIBLE_DEVICES="0" python tools/trainval.py configs/trainval/tinaface/tinaface_r50_fpn_bn.py

conda deactivate

