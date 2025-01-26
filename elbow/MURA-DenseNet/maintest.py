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

stringNegative = 'negative'
stringPositive = 'positive'

batch_size = 1


device = torch.device('cuda:0')

model = models.densenet169(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 1)
model = torch.nn.DataParallel(model).cuda()

print(set)
for modelname in ['raid5-fix-1024']:
                  #'raid5-fix-224',
                  #'crop2-fix-224']:
                  #'crop2-fix-512']:
    if 'crop2-fix-224' in modelname:
        imageData, imageDataloaders, typ = data.getData('crop2-fix', dataset, modelname, [set], batch_size)
    elif 'crop2-fix-512' in modelname:
        imageData, imageDataloaders, typ = data.getData('crop512-megafix', dataset, modelname, [set], batch_size)
    else:
        imageData, imageDataloaders, typ = data.getData('combinedtest-study', dataset, modelname, [set], batch_size)
    for metric in ['loss', 'acc']:
        #bestacc = 0
        #besti = -1
        #bestm = ''
        #acc = 0
        #bestauc = 0
        for i in range(1,28):
            #print(i)
            #print(str(i))
            print(str(modelname) + '-' + str(i) + '-' + str(metric))
            test.loadandtest('./models/' + str(modelname) + '-' + str(i) + '_' + str(metric) + '.pth', model, imageDataloaders[set], typ)
            #print('-'*10)
            #if acc >= bestacc:
            #    if auc >= bestauc:
            #        bestacc = acc
            #        bestauc = auc
            #        besti = i
            #        bestm = metric

        #print('Best model: ' + str(besti) + ' _ ' + str(bestacc) + ' _ ' + str(bestauc) + ' _ ' + str(bestm))



