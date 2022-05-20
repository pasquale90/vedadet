import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
import os


def writeJson(data,path):
    with open(path,'w') as fp:
        json.dump(data,fp,indent=1)

################################################################# - Detection model - ################################################
import tools.infer as tfinference
from vedacore.parallel import collate, scatter

class TinaFace_model(nn.Module):
    def __init__(self):
        super().__init__()
        print("Constructor")
    
    #def Tinaface_Init(self,device,weights):
    def Tinaface_Init(self,device):
        from vedacore.misc import Config, color_val, load_weights

        if device=='cuda:0':
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"]="0"

        self.cfg = Config.fromfile('configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py')
        self.engine, self.data_pipeline, self.device = tfinference.prepare(self.cfg)
    
    import os
    
    #def anomalyInfer(model, input, outputPath, device):
    def TinafaceInfer(self,input,outputPath):
        
        toRet = []
        for image in input:
            print(image)
            imagePath=image['path']
            imageID=image['id']

            class_names = self.cfg.class_names

            data = dict(img_info=dict(filename=imagePath), img_prefix=None)
            data = self.data_pipeline(data)
            data = collate([data], samples_per_gpu=1)
        
            if self.device != 'cpu':
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                # just get the actual data from DataContainer
                data['img_metas'] = data['img_metas'][0].data
                data['img'] = data['img'][0].data
            result = self.engine.infer(data['img'], data['img_metas'])[0]

            insects_count=len(result[0])
            print("Insects count ",insects_count)

            imsavepath=os.path.join(outputPath,imagePath.split('/')[-1])
            tfinference.plot_result(result, imagePath, class_names, imsavepath)
            temp = {'id': str(imageID), 'name':"image_"+str(imageID)+".png", "result": insects_count}
            toRet.append(temp)
        writeJson(toRet, os.path.join(outputPath,'count.json'))
        return toRet

