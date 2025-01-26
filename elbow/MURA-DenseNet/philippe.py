from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import logging
from torch.autograd import Variable

from data import getData
import csv

if __name__ == "__main__":

    logging.basicConfig(filename='log.log', level=logging.INFO)

    set = 'test'
    dataset = 'RAID'

    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model = models.densenet169(pretrained=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)
    model = torch.nn.DataParallel(model).cuda()

    models = ['raid5-512-8_acc.pth', 'raid5-fix-1024-12_loss.pth', 'crop2-fix-512-7_acc.pth']

    folders = ['test', 'valid', 'train', 'combinedtest']
    study_type = ['study', 'radio']

    for study in study_type:
        for folder in folders:
            if folder != 'combinedtest' or study != 'radio':
                for z, modelfname in enumerate(models):
                    with open('results_' + str(folder) + '_' + str(study) + '_' + str(z) + '.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)

                        writer.writerow(['ID', 'Label', modelfname])
                        if folder == 'test' or folder == 'combinedtest':
                            set = 'test'
                        else:
                            set = folder
                        
                        print("Evaluating " + str(modelfname) + " for " + str(study) + " " + str(folder) + ".")
                        if folder == 'combinedtest':
                            if 'crop' in modelfname:
                                imageData, imageDataloaders, type = getData('combinedtest-crop2-study', dataset, modelfname, [set], batch_size)
                            else:
                                imageData, imageDataloaders, type = getData('combinedtest-study', dataset, modelfname, [set], batch_size)
                        else:
                            if study == 'study':
                                if 'crop2-fix-224' in modelfname or 'crop2-224' in modelfname:
                                    imageData, imageDataloaders, type = getData('crop2-study', dataset, modelfname, [set], batch_size)
                                elif 'crop2-fix-512' in modelfname or 'crop2-512' in modelfname:
                                    imageData, imageDataloaders, type = getData('crop512-study', dataset, modelfname, [set], batch_size)
                                else:
                                    imageData, imageDataloaders, type = getData('raid5-study', dataset, modelfname, [set], batch_size)
                            else:
                                if 'crop2-fix-224' in modelfname or 'crop2-224' in modelfname:
                                    imageData, imageDataloaders, type = getData('crop2-fix', dataset, modelfname, [set], batch_size)
                                elif 'crop2-fix-512' in modelfname or 'crop2-512' in modelfname:
                                    imageData, imageDataloaders, type = getData('crop512-fix', dataset, modelfname, [set], batch_size)
                                else:
                                    imageData, imageDataloaders, type = getData('raid5-fix', dataset, modelfname, [set], batch_size)


                        if 'fix' in modelfname:
                            path = './savedModels/' + modelfname
                        else:
                            path = './savedOldModels/' + modelfname

                        model.load_state_dict(torch.load(path, map_location=device))
                        model.eval()

                        if study == 'study':
                            with torch.no_grad():
                                for i, data in enumerate(imageDataloaders[set]):
                                    print(i, end="\r")
                                    inputs = data['images'][0].to(device)
                                    labels = data['label'].type(torch.cuda.FloatTensor).to(device)
                                    outputs = torch.t(model(inputs))
                                    label = labels.item()
                                    pred = (torch.sigmoid(torch.max(outputs.data))).type(torch.cuda.FloatTensor).item() 
                                    writer.writerow([data['link'], label, pred])
                        else:
                            with torch.no_grad():
                                for i, data in enumerate(imageDataloaders[set]):
                                    print(i, end="\r")
                                    inputs = data['images'].to(device)
                                    labels = data['label'].type(torch.FloatTensor).to(device)
                                    outputs = torch.t(model(inputs))[0]
                                    label = labels.item()
                                    pred = (torch.sigmoid(torch.max(outputs.data))).type(torch.cuda.FloatTensor).item()
                                    writer.writerow([data['link'], label, pred])
