# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

##############################################
# Import Library
##############################################

from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights, vgg19, VGG19_Weights, resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import bernoulli, zscore
from tqdm.notebook import tqdm

import calibration_metric
import importlib
importlib.reload(calibration_metric)
from calibration_metric import ece, ace, tce



##############################################
# Prepare Dataset
##############################################

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
imagenet_data = torchvision.datasets.ImageNet('./imagenet_root/', split='val', transform=imagenet_transform)
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=128, shuffle=True)



##############################################
# Prepare Model
##############################################

model_01 = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model_02 = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
model_03 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_04 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_05 = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

models = [ model_01, model_02, model_03, model_04, model_05 ]



##############################################
# Predict by Model
##############################################

def run_test(models, loader, label_ids):
    [ model.eval() for model in models ]

    labels = np.array([]).astype(int)
    preds = [ np.array([]) for model in models ]
    
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader)):
            labels = np.r_[ labels, ( torch.isin(y, label_ids) ).detach().numpy().astype(int) ]
            for j in range(len(models)): 
                preds[j] = np.r_[ preds[j], F.softmax(models[j](x), dim=1)[:,label_ids].sum().detach().numpy() ]
    
    return labels, preds

labels, preds = run_test(models, data_loader, torch.arange(151, 276))

labels.tofile('./labels.npy')
[ preds[j].tofile('./preds_0'+str(j)+'.npy') for j in range(len(preds)) ]



