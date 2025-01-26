from __future__ import division
from __future__ import print_function

import copy
import time
import logging
import torchvision
import cv2
import numpy as np

from earlystopping import EarlyStopping

import torch
from torch.autograd import Variable


def train_model(model, string, dataloaders, type, criterion, optimizer, scheduler, size, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000000000.0

    es = EarlyStopping(patience = 40)

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            preds_1 = 0.0
            labels_1 = 0.0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')

                if type == 'study':
                    inputs = data['images'][0]
                else:
                    inputs = data['images']
                labels = data['label'].type(torch.FloatTensor)
                #torchvision.utils.save_image(data['images'][0], 'out/output_' + str(epoch) + '_' +  str(phase) + '_' + str(i) + '.png')
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if type == 'study':
                        outputs = torch.max(torch.t(model(inputs))).unsqueeze(0)
                    else:
                        outputs = torch.t(model(inputs))[0]
                    loss = criterion(outputs, labels.data)
                    preds = (torch.sigmoid(outputs.data) > 0.5).type(torch.cuda.FloatTensor)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_1 += torch.sum(preds)
                labels_1 += torch.sum(labels.data)


            epoch_loss = running_loss / size[phase]
            epoch_acc = running_corrects.double() / size[phase]

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} / {}'.format(preds_1, labels_1))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} / {}'.format(preds_1, labels_1))


            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), './models/' + string + '_acc.pth')
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), './models/' + string + '_loss.pth')
                val_acc_history.append(epoch_acc)

        if es.step(epoch_loss):
            break

        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
