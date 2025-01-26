import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
import torchvision
from PIL import Image, ImageOps
from tqdm import tqdm
import logging

import cv2

def prepareAugmentation(studyType, modelfname):
    if '512' in studyType:
        size = (512, 512)
    else:
        if '512' in modelfname:
            size = (512, 512)
        elif '800' in modelfname:
            size = (800, 800)
        elif '1024' in modelfname:
            size = (1024, 1024)
        else:
            size = (224, 224)
    #print(size)
    dataAugmentation = {
        'train': transforms.Compose([
            #transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            #transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return dataAugmentation

def getData(studyType, dataset, modelfname, sets, batch):
    data = {}
    batch_size = batch
    for phase in sets:
        i = 0
        data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        if studyType.endswith('study'):
            data, i = handleStudyData(data, phase, i, studyType, dataset)
            type = 'study'
            batch_size = 1
        elif studyType.startswith('crop') or studyType.endswith('fix'):
            data, i = handleCropData(data, phase, i, studyType, dataset)
            type = 'crop'
        else:
            data, i = handleImageData(data, phase, i, studyType, dataset)
            type = 'image'
    dataAugmentation = prepareAugmentation(studyType, modelfname)
    if type == 'study':
        image_datasets = {x: StudyData(data[x], dataAugmentation[x]) for x in sets}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers=1) for x in sets}
    elif type == 'crop':
        image_datasets = {x: ImageData(data[x], dataAugmentation[x]) for x in sets}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers=8) for x in sets}
    else:
        image_datasets = {x: ImageData(data[x], dataAugmentation[x]) for x in sets}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size, shuffle=True, num_workers=8) for x in sets}
    return data, dataloaders, type

def handleStudyData(data, phase, i, studyType, dataset):
    study_label = {'positive': 1, 'negative': 0}
    BASE_DIR = dataset + '/%s/%s/' % (phase, studyType)
    patients = list(os.walk(BASE_DIR))[0][1]
    for patient in patients:
        for study in os.listdir(BASE_DIR + patient):
            try:
                label = study_label[study.split('_')[1]]
            except Exception as e:
                print(BASE_DIR)
                print(patient)
                print(study)
            path = BASE_DIR + patient + '/' + study + '/'
            count = len(os.listdir(path))
            data[phase].loc[i] = [path, count, label]
            i += 1
    print('-'*10)
    return data, i

def handleImageData(data, phase, i, studyType, dataset):
    study_label = {'positive': 1, 'negative': 0}
    BASE_DIR = dataset + '/%s/%s/' % (phase, studyType)
    patients = list(os.walk(BASE_DIR))[0][1]
    for patient in tqdm(patients):
        for study in os.listdir(BASE_DIR + patient):
            label = study_label[study.split('_')[1]]
            path = BASE_DIR + patient + '/' + study
            images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for image in images:
                data[phase].loc[i] = [path + '/' + image, 1, label]
                i += 1
    return data, i

def handleCropData(data, phase, i, studyType, dataset):
    study_label = {'P': 1, 'N': 0}
    BASE_DIR = dataset + '/%s/%s/' % (phase, studyType)
    for xray in os.listdir(BASE_DIR):
        label = study_label[xray.split('_')[0]]
        path = BASE_DIR
        data[phase].loc[i] = [path + '/' + xray, 1, label]
        i += 1
    return data, i


class StudyData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        try:
            for f in os.listdir(path):
                image = Image.open(path + f)
                image = self.transform(image).type(torch.FloatTensor)
                image = 1 - image
                images.append(image)
        except:
            pass
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label, 'link': path}
        return sample


class ImageData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx, 0]
        images = pil_loader(path)
        images = self.transform(images)
        images = 1 - images
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label, 'link': path}
        return sample
