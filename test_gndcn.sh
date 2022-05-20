#!/bin/bash
source /home/melissap/miniconda3/etc/profile.d/conda.sh
conda activate tinaface3

dt=`date "+%Y%m%d%H%M"`

for dol in data/WIDERFace/WIDER_test/images/*
do
    echo -e "dol ${dol}"

    dolname=$(basename "$dol")
    echo dolname $dolname

    for im in ${dol}/*
    do
        echo "image ${im}"
        CUDA_VISIBLE_DEVICES="0" python tools/infer.py configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py ${im}
        filename=$(basename "$im")
        savepath=testing_fork/testing_results/${dt}_blur_and_clean/${dolname%%/}/
        mkdir $savepath
        echo savepath $savepath
        mv out.jpg ${savepath}${filename} #$savepath
        #break
    done

    #if [[ -f $file ]]; then
        #copy stuff ....
    #fi
#break
done
echo "done with testing"

#for number in {0..515}
#do 
#echo "Validating the model weights at epoch ${number}"
#CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/trainval/tinaface/tinaface_r50_fpn_bn.py /media/melissap/AA761F36761F032D/WINPART/CoRoSect/7.experiments/7.TinaFace/tinaface/vedadet/workdir/tinaface_r50_fpn_bn/epoch_${number}_weights.pth
#python test{$number}.py
#done
#exit 0

#conda deactivate

