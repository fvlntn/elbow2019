from __future__ import division
from __future__ import print_function

import copy
import time
import logging

import torch
import torchvision
import torchvision.transforms as transforms
from scipy.optimize import differential_evolution
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import random


def calculate_auc(labels, preds):
    fpr, tpr, _ = metrics.roc_curve(np.array(labels), np.array(preds), pos_label=1)
    return metrics.auc(fpr, tpr)


def show(acc, auc, weights):
    print('Acc: ' + str(acc) + ' ---- AUC: ' + str(auc) + ' ---- Weights: ' + str(weights))

def showaccauc(acc, auc):
    print('Acc: ' + str(acc) + ' ---- AUC: ' + str(auc) + ' ----')



def shuffled_range(num):
    r = list(range(num))
    random.shuffle(r)
    return r


def normalize(weights):
    result = np.linalg.norm(weights, 1)
    if result == 0.0:
        return weights
    return weights / result


def get_accuracy_auc(weights, labels, preds, links):
    confidence = 0.5
    true0 = 0.0
    true1 = 0.0
    false0 = 0.0
    false1 = 0.0
    total = len(preds)
    ensemblepreds = []
    for z in range(total):
        finalpred = 0
        totalweights = 0
        for k in range(len(preds[z])):
            finalpred += preds[z][k] * weights[k]
            totalweights += weights[k]
        finalpred /= totalweights
        ensemblepreds.append(finalpred)
        if finalpred <= confidence and labels[z] == 0:
            true0 += 1
        elif confidence <= finalpred and labels[z] == 1:
            true1 += 1
        elif finalpred <= confidence and labels[z] == 1:
            false0 += 1
            #print('FN ' + str(links[z]))
        elif confidence <= finalpred and labels[z] == 0:
            false1 += 1
            #print('FP ' + str(links[z]))
    #print('FP:' + str(false1))
    #print('FN:' + str(false0))
    acc = (true0 + true1) / total
    auc = calculate_auc(labels, ensemblepreds)
    return acc, auc

def test_image_model(model, testloader1, confidence):
    total = 0
    true0 = 0.0
    true1 = 0.0
    false0 = 0.0
    false1 = 0.0
    meandiff = 0.0

    totalpreds = []
    totallabels = []

    with torch.no_grad():
        for i, data in enumerate(testloader1):
            print(i, end='\r')
            total += 1
            inputs = data['images'].cuda()
            labels = data['label'].type(torch.FloatTensor).cuda()
            totallabels.append(labels.item())
            outputs = torch.t(model(inputs))[0]
            totalpreds.append((torch.sigmoid(torch.max(outputs.data))).type(torch.cuda.FloatTensor).item())
            torchvision.utils.save_image(data['images'][0], 'out/output_' + str(i) + '.png')
            if torch.sigmoid(outputs.data)[0].item() <= confidence and labels[0] == 0:
                true0 += 1
                meandiff += torch.sigmoid(outputs.data)[0].item()
                #print('TN,' + str(data['link']))
            elif confidence <= torch.sigmoid(outputs.data)[0].item() and labels[0] == 1:
                true1 += 1
                meandiff += 1 - torch.sigmoid(outputs.data)[0].item()
                #print('TP,' + str(data['link']))
            elif torch.sigmoid(outputs.data)[0].item() <= confidence and labels[0] == 1:
                false0 += 1
                meandiff += 1 - torch.sigmoid(outputs.data)[0].item()
                print('FN,' + str(data['link']))
            elif confidence <= torch.sigmoid(outputs.data)[0].item() and labels[0] == 0:
                false1 += 1
                meandiff += torch.sigmoid(outputs.data)[0].item()
                print('FP,' + str(data['link']))
    correct = true0 + true1
    acc = 100 * correct / total
    auc = calculate_auc(totallabels, totalpreds)
    # print("F : " + str(false0+false1))
    print("TN : " + str(true0))
    print("TP : " + str(true1))
    print("FN : " + str(false0))
    print("FP : " + str(false1))
    #print("Mean diff : " + str(meandiff))
    #print('TPR : ' + str((true1/(true1+false0))))
    #print('FPR : ' + str((false1/(false1+true0))))
    #print('Accuracy: ' + str(acc))
    #print('Guessed: ' + str(100 * total2 / total))
    # print('Total: ' + str(total))
    # print('-' * 10)
    showaccauc(acc,auc)
    return acc, auc

def test_image_model_on_study(model, testloader1, confidence):
    total = 0
    true0 = 0.0
    true1 = 0.0
    false0 = 0.0
    false1 = 0.0
    totallabels = []
    totalpreds = []
    with torch.no_grad():
        for i, data in enumerate(testloader1):
            print(i, end='\r')
            total += 1
            inputs = data['images'][0].cuda()
            labels = data['label'].type(torch.FloatTensor).cuda()
            torchvision.utils.save_image(data['images'][0], 'out/outputstudy_' + str(i) + '.png')
            outputs = torch.t(model(inputs))
            totallabels.append(labels.item())
            totalpreds.append((torch.sigmoid(torch.max(outputs.data))).type(torch.cuda.FloatTensor).item())
            if torch.sigmoid(torch.max(outputs.data)).item() <= confidence and labels[0] == 0:
                true0 += 1
                #print('TN,' + str(data['link']))
            elif confidence <= torch.sigmoid(torch.max(outputs.data)).item() and labels[0] == 1:
                true1 += 1
                #print('TP,' + str(data['link']))
            elif torch.sigmoid(torch.max(outputs.data)).item() <= confidence and labels[0] == 1:
                false0 += 1
                #print('FN,' + str(data['link']))
            elif confidence <= torch.sigmoid(torch.max(outputs.data)).item() and labels[0] == 0:
                false1 += 1
                #print('FP,' + str(data['link']))
    correct = true0 + true1
    acc = 100 * correct / total
    auc = calculate_auc(totallabels,totalpreds)
    # print("F : " + str(false0 + false1))
    print("TN : " + str(true0))
    print("TP : " + str(true1))
    print("FN : " + str(false0))
    print("FP : " + str(false1))
    #print('Accuracy: ' + str(acc))
    # print('Total: ' + str(total))
    # print('-' * 10)
    showaccauc(acc,auc)
    return acc, auc

def get_preds_ensemble(models, testloaders):
    its = []

    for testloader in testloaders:
        its.append(iter(testloader))
    totalpreds = []
    totallabels = []
    links = []

    with torch.no_grad():
        for i in range(len(testloaders[0].dataset)):
            print(i, end='\r')
            inputs = []
            preds = []
            firstlink = ''
            for it in its:
                data = next(it)
                #print(data['link'])
                #print('-'*10)
                #if firstlink == '':
                #    firstlink = data['link'][0].split('patient')[1]
                #else:
                #    while data['link'][0].split('patient')[1] != firstlink:
                #        data = next(it)
                inputs.append(data['images'][0].cuda())
                input = data['images'][0].cuda()
                links.append(data['link'][0].split('patient')[1])
                label = data['label'].type(torch.FloatTensor).cuda().item()
            for j in range(len(models)):
                output = torch.t(models[j](input))
                pred = (torch.sigmoid(torch.max(output.data))).type(torch.cuda.FloatTensor)
                preds.append(pred.item())
            totalpreds.append(preds)
            totallabels.append(label)
    return totalpreds, totallabels, links


def test_loss(weights, labels, preds):
    normalized = normalize(weights)
    acc, auc = get_accuracy_auc(normalized, labels, preds)
    return 2.0 - acc - auc


def test_ensemble_model(models, testloaders):
    preds, labels, links = get_preds_ensemble(models, testloaders)
    weights = [1/len(models) for _ in range(len(models))]
    #acc, auc = get_accuracy_auc(weights, labels, preds, links)
    #show(acc, auc, weights)
    bound_w = [(0.0, 1.0) for _ in range(len(models))]
    search_arg = (labels, preds)
    bestloss = 0
    #for i in range(0, 50000):
        #result = differential_evolution(test_loss, bound_w, search_arg, maxiter=10000000, tol=1e-10)
        #weights = normalize(result['x'])
    bestweights = ''
    bestauc = 0
    maxrange = 10
    for i in range(0,maxrange):
        print(i, end='\r')
        weightss = [[i/maxrange, 1-i/maxrange]]
        for weights in weightss:
            acc, auc = get_accuracy_auc(weights, labels, preds, links)
            loss = acc
            #show(acc,auc,weights)
            if loss >= bestloss:
                #show(acc,auc,weights)
                bestloss = loss
                show(acc,auc,weights)
                #if loss > bestloss:
                #    bestauc = 0
                #if auc >= bestauc:
                #    bestweights = weights
                #    bestauc = auc
                #    show(acc, auc, weights)
    print('-'*10)
    print(bestweights)
    return acc, auc


def loadandtest(path, model, testloader1, type):
    model = load(path, model)
    if type == 'study':
        acc, auc = test_image_model_on_study(model, testloader1, 0.5)
    else:
        acc, auc = test_image_model(model, testloader1, 0.5)
    return acc


def load(path, model):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
