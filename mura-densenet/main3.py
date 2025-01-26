from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import logging
from torch.autograd import Variable

import data
import test

import numpy as np
import train

print("PyTorch Version: ", torch.__version__)

logging.basicConfig(filename='log.log', level=logging.INFO)

set = 'test'
dataset = 'RAID'

batch_size = 1

device = torch.device('cuda:0')

model = models.densenet169(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)
model = torch.nn.DataParallel(model).cuda()


raidmodels = [ 'raid5-fix-224-10_loss.pth',
                    'raid5-fix-224-14_acc.pth',
                    'raid5-fix-224-22_loss.pth',
                    'raid5-fix-224-32_loss.pth',
                    'raid5-fix-224-37_acc.pth',
                    'raid5-fix-224-5_acc.pth',
                    'raid5-fix-512-1_acc.pth',
                    'raid5-fix-512-1_loss.pth',
                    'raid5-fix-512-2_acc.pth',
                    'raid5-fix-512-32_acc.pth',
                    'raid5-fix-512-36_loss.pth',
                    'raid5-fix-512-39_loss.pth',
                    'raid5-224-1_acc.pth',
                    'raid5-512-8_acc.pth',
                    'raid5-800-5_loss.pth',
                    'raid5-fix-1024-12_loss.pth',
                    'raid5-fix-1024-2_loss.pth'
            ]
cropmodels = [ 'crop2-fix-224-14_acc.pth',
        'crop2-fix-224-2_loss.pth',
        'crop2-fix-224-32_acc.pth',
        'crop2-fix-224-36_loss.pth',
        'crop2-fix-224-40_loss.pth',
        'crop2-fix-224-7_acc.pth',
        'crop2-fix-512-14_loss.pth',
        'crop2-fix-512-1_acc.pth',
        'crop2-fix-512-32_acc.pth',
        'crop2-fix-512-32_loss.pth',
        'crop2-fix-512-37_acc.pth',
        'crop2-fix-512-37_loss.pth',
        'crop2-fix-512-7_acc.pth',
        'crop2-224-32_loss.pth',
        'crop2-512-3_loss.pth'
        ]
for modelfname in cropmodels:
    print(str(modelfname))
    if 'crop2-fix-224' in modelfname or 'crop2-224' in modelfname:
        imageData, imageDataloaders, type = data.getData('newtest-crop2-study', dataset, modelfname, [set], batch_size)
    elif 'crop2-fix-512' in modelfname or 'crop2-512' in modelfname:
        imageData, imageDataloaders, type = data.getData('newtest-crop2-study', dataset, modelfname, [set], batch_size)
    else:
        if 'fix' in modelfname:
            imageData, imageDataloaders, type = data.getData('combinedtest-study', dataset, modelfname, [set], batch_size)
        else:
            imageData, imageDataloaders, type = data.getData('combinedtest-study', dataset, modelfname, [set], batch_size)
    try:   
        acc = test.loadandtest('./savedOldModels/' + modelfname, model, imageDataloaders[set], type)
    except Exception as e:
        acc = test.loadandtest('./savedModels/' + modelfname, model, imageDataloaders[set], type)
    print('-'*10)
