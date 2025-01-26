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

# model1 = models.densenet169(pretrained=True)
# num_features = model1.classifier.in_features
# model1.classifier = nn.Linear(num_features, 1)
# model1 = torch.nn.DataParallel(model1).cuda()
#
model2 = models.densenet169(pretrained=True)
num_features2 = model2.classifier.in_features
model2.classifier = nn.Linear(num_features2, 1)
model2 = torch.nn.DataParallel(model2).cuda()

model3 = models.densenet169(pretrained=True)
num_features3 = model3.classifier.in_features
model3.classifier = nn.Linear(num_features3, 1)
model3 = torch.nn.DataParallel(model3).cuda()

model4 = models.densenet169(pretrained=True)
num_features4 = model4.classifier.in_features
model4.classifier = nn.Linear(num_features4, 1)
model4 = torch.nn.DataParallel(model4).cuda()

#model5 = models.densenet169(pretrained=True)
#num_features5 = model5.classifier.in_features
#model5.classifier = nn.Linear(num_features5, 1)
#model5 = torch.nn.DataParallel(model5).cuda()

#modelfnames = ['raid5-224-1_acc', 'raid5-512-8_acc', 'raid5-800-5_loss']
#modelstudies = ['raid5-study', 'raid5-study', 'raid5-study']

# imageData1, imageDataloaders1, type1 = data.getData('crop2-study', dataset, 'crop2-224-32-loss', [set], batch_size)
imageData2, imageDataloaders2, type2 = data.getData('combinedtest-crop2-study', dataset, 'crop2-fix-512-7_acc', [set], batch_size)
imageData3, imageDataloaders3, type3 = data.getData('combinedtest-study', dataset, 'raid5-fix-1024-12_loss', [set], batch_size)
imageData4, imageDataloaders4, type4 = data.getData('combinedtest-study', dataset, 'raid5-512-8_acc', [set], batch_size)
#imageData5, imageDataloaders5, type5 = data.getData('combinedtest-study', dataset, 'raid5-800-5_loss', [set], batch_size)

# model1 = test.load('./savedOldModels/' + 'crop2-224-32-loss' + '.pth', model1)
model2 = test.load('./savedModels/' + 'crop2-fix-512-7_acc' + '.pth', model2)
model3 = test.load('./savedModels/' + 'raid5-fix-1024-12_loss' + '.pth', model3)
model4 = test.load('./savedOldModels/' + 'raid5-512-8_acc' + '.pth', model4)
#model5 = test.load('./savedOldModels/' + 'raid5-800-5_loss' + '.pth', model5)

modelsList = [model2, model3, model4]
dlList = [imageDataloaders2[set], imageDataloaders3[set], imageDataloaders4[set]]
acc, auc = test.test_ensemble_model(modelsList, dlList)

# modelsList = [model1, model2, model3, model4, model5]
# dlList = [imageDataloaders1[set], imageDataloaders2[set], imageDataloaders3[set], imageDataloaders4[set], imageDataloaders5[set]]
# acc, auc = test.test_ensemble_model(modelsList, dlList)
