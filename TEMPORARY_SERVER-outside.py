import os
import sys
import subprocess

pwd=os.getcwd()
#give path to the vedadet_ folder
project_dir= "/home/melissap/Desktop/TinaFace-integration/detection_submodule/vedadet_" #os.path.join(pwd,'vedadet_')
sys.path.insert(0,project_dir)
os.chdir(project_dir)
import TF_utils as utils

input=[{'path':'data/WIDERFace/WIDER_test/images/37/37_040159.jpg', 'id':0}, 
        {'path':'data/WIDERFace/WIDER_test/images/30/30_040169.jpg', 'id':1}]

def main():
    print(os.getcwd())
    print(os.listdir)

    
    subprocess.run('bash activate_conda.sh', shell=True)

    tfmodel=utils.TinaFace_model()

    tfmodel.Tinaface_Init('cuda:0')

    tfmodel.TinafaceInfer(input,outputPath='VI_results')

    os.chdir(pwd)


if __name__=="__main__":
    main()