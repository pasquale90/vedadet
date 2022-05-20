#!/bin/bash
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate tinaface2

for number in {1..256}
do 
echo "Running validation for the model saved at ${number} epoch"
CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/trainval/tinaface/tinaface_r50_fpn_gn_dcn.py workdir/tinaface_r50_fpn_gn_dcn/epoch_${number}_weights.pth
#python test{$number}.py
done
exit 0
cp nohup.out testing_folder/tfdcn/256epochs/vallog.out
conda deactivate

