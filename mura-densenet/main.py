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
import train

print("PyTorch Version: ", torch.__version__)
logging.basicConfig(filename='logcrop224.log', level=logging.INFO)
# torch.manual_seed(1498920)
# torch.cuda.manual_seed(1498920)
# torch.backends.cudnn.deterministic = True

studyType = 'raid5-fix'
modelName = 'raid5-fix-1024-'

dataset = 'RAID'
sets = ['train', 'valid']
batch_size = 8
imageData, imageDataloaders, type = data.getData(studyType, dataset, modelName, sets, batch_size)

if type == 'crop':
    stringNegative = 'N_'
    stringPositive = 'P_'
else:
    stringNegative = 'negative'
    stringPositive = 'positive'

normalImages = {x: imageData[x][imageData[x]['Path'].str.contains(stringNegative)]['Count'].sum() for x in sets}
abnormalImages = {x: imageData[x][imageData[x]['Path'].str.contains(stringPositive)]['Count'].sum() for x in sets}

size = {x: len(imageData[x]) for x in sets}
print(normalImages)
print(abnormalImages)
weight = torch.tensor([normalImages['train']/abnormalImages['train']])
print('Weight 0/1: {:4f}'.format(weight.item()))

lrlist = [0.00005, 0.00003, 0.00001, 0.000005, 0.000003, 0.000001]
patiencelist = [5, 10, 15, 20, 25]
i = 0
for lr in lrlist:
    for patience in patiencelist:
        i = i+1
        print('LR = ' + str(lr))
        model = models.densenet169(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)
        model = torch.nn.DataParallel(model).cuda()

        num_epochs = 200
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=patience, verbose=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight.cuda())
        patiencelist = [5, 10, 15, 20, 25]cc = test.loadandtest('./savedOldModels/' + modelfname, model, imageDataloaders[set], 0.5, type)
        except Exception as e:
            acc = test.loadandtest('./savedModels/' + modelfname, model, imageDataloaders[set], 0.5, type)
        print('-'*10)       model, hist = train.train_model(model, modelName + str(i), imageDataloaders, type, criterion, optimizer, scheduler, size, num_epochs)
