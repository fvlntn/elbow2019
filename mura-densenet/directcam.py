# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
# import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import cv2
import torchvision
import data
import pdb
import csv

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def returnCAM2(feature_conv, weight_softmax, class_idx, min, max):
    size_upsample = (512, 512)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - min
        cam_img = cam / ( max - min ) 
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (512, 512)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        print(np.min(cam))
        print(np.max(cam))
        print('-'*10)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        #cam_img = cv2.bitwise_not(cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':
    true0 = 0
    true1 = 0
    false0 = 0
    false1 = 0
    idk = 0
    conf = 0.0

    batch_size = 1
    dataset = 'RAID'
    set = 'test'
    modelfname = 'crop2-fix-512-7_acc.pth'
    imageData, imageDataloaders, type = data.getData('crop2-fix', dataset, modelfname, [set], batch_size)
    #min, max = getMinMax(modelfname, imageData, imageDataloaders)

    min = 99999
    max = -99999
    for i, data in enumerate(imageDataloaders['test']):
        img = data['images']
        net = models.densenet169(pretrained=True)
        num_features = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_features, 1)
        model = torch.nn.DataParallel(net).cuda()
        model.load_state_dict(torch.load('./savedModels/' + str(modelfname)))
        finalconv_name = 'features'
        net.eval()
        features_blobs = []
        net._modules.get(finalconv_name).register_forward_hook(hook_feature)
        params = list(net.parameters())
        weight_softmax = params[-2].data.cpu().numpy()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = data['images']
        img_tensor2 = data['images']
        label = data['label']
        img_variable = Variable(img_tensor.cuda())
        classes = ['abnormal']
        logit = net(img_variable)
        h_x = torch.sigmoid(torch.t(logit)[0].data).type(torch.cuda.FloatTensor)
        #h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        #CAMs = returnCAM(features_blobs[0], weight_softmax, [idx])
        bz, nc, h, w = features_blobs[0].shape
        cam = weight_softmax[idx].dot(features_blobs[0].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        maxC = np.max(cam)
        #print(maxC)
        minC = np.min(cam)
        #print(minC)
        #print('-'*10)
        if(maxC >= max):
            max = maxC
        if(minC <= min):
            min = minC
    print(min)
    print(max)

    print('----------')
    for i, data in enumerate(imageDataloaders['test']):
        net = models.densenet169(pretrained=True)
        num_features = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_features, 1)
        model = torch.nn.DataParallel(net).cuda()
        model.load_state_dict(torch.load('./savedModels/' + str(modelfname)))
        finalconv_name = 'features'

        net.eval()

        features_blobs = []

        net._modules.get(finalconv_name).register_forward_hook(hook_feature)

        params = list(net.parameters())
        weight_softmax = params[-2].data.cpu().numpy()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])


        img_tensor = data['images']

        img_tensor2 = data['images']
        label = data['label']
        img_variable = Variable(img_tensor.cuda())
        classes = ['abnormal']
        logit = net(img_variable)
        h_x = torch.sigmoid(torch.t(logit)[0].data).type(torch.cuda.FloatTensor)
        #h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        print(probs)
        torchvision.utils.save_image(img_tensor2, 'CAM/test_' + str(i) + '.jpg')
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx])
        #CAMs = returnCAM2(features_blobs[0], weight_softmax, [idx], min, max)
        print(data['link'][0].split('/')[4])
        if (probs <= 0.5 - conf and label.item() == 0):
            true0 += 1
            suffix = 'TN'
        elif (probs > 0.5 + conf and label.item() == 1):
            true1 += 1
            suffix = 'TP'
        elif (probs <= 0.5 - conf and label.item() == 1):
            false0 += 1
            suffix = 'FN'
        elif (probs > 0.5 + conf and label.item() == 0):
            false1 += 1
            suffix = 'FP'
        else:
            idk += 1
            suffix = 'IDK'
        name = 'CAM/' + suffix + '_' + str((data['link'][0].split('/')[4].split('.')[0])) + '.jpg'
        img = cv2.imread('CAM/test_' + str(i) + '.jpg')
        img = cv2.bitwise_not(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print(data['link'][0])
        #cv2.imwrite('CAM/' + str((data['link'][0].split('/')[4].split('.')[0])) + '.jpg', img)        
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        with open('cam.csv', mode='a') as cam_file:
            cam = csv.writer(cam_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            cam.writerow([np.min(CAMs[0]),np.max(CAMs[0]),probs])
        print(np.max(CAMs[0]))
        print(np.min(CAMs[0]))
        print(heatmap.shape)
        result = heatmap * 0.3 + img * 0.5
        cv2.putText(result, 'Prediction: ' + str(probs[0]) + ' | Truth : ' + str(label.item()) + ' | Max : ' + str(np.max(CAMs[0])) + ' | Min : ' + str(np.min(CAMs[0])), (5,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imwrite(name, result)
    print("TP : " + str(true1))
    print("TN : " + str(true0))
    print("FP : " + str(false1))
    print("FN : " + str(false0))
    print("IDK : " + str(idk))
