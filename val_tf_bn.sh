#!/bin/bash
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate tinaface2

for number in {0..515}
do 
echo "Validating the model weights at epoch ${number}"
CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/trainval/tinaface/tinaface_r50_fpn_bn.py /media/melissap/AA761F36761F032D/WINPART/CoRoSect/7.experiments/7.TinaFace/tinaface/vedadet/workdir/tinaface_r50_fpn_bn/epoch_${number}_weights.pth
#python test{$number}.py
done
exit 0

conda deactivate

