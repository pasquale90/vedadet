import TF_utils as utils
import sys


input=[{'path':'data/WIDERFace/WIDER_test/images/37/37_040159.jpg', 'id':0}]

def main():

    tfmodel=utils.TinaFace_model()

    tfmodel.Tinaface_Init('cuda:0')
    

    tfmodel.TinafaceInfer(input,outputPath='VI_results')
   


if __name__=="__main__":
    main()