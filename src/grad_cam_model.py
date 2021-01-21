import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

import os
import numpy as np
import torch
from PIL import Image

#from src.grad_cam import *
from grad_cam import *

def run_models(input_path, cam_img, gb_dir, cam_dir):
        print("starting model building")
        model = models.resnet50(pretrained=True)
        print("generated resnet model")
        grad_cam = GradCam(model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=False)

        img = cv2.imread(input_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input_img = preprocess_image(img)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index = None
        print("starting GRAD-CAM")
        mask = grad_cam(input_img, target_index)

        show_cam_on_image(img, mask, cam_img)

        print("starting GUIDED GRAD-CAM")
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
        #print(model._modules.items())
        gb = gb_model(input_img, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        gb = deprocess_image(gb)

        print("writing out images")
        print("gbdir: ", gb_dir)
        cv2.imwrite(gb_dir, gb)
        cv2.imwrite(cam_dir, cam_gb)
        return