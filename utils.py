from __future__ import print_function, division

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2

import pandas as pd
from skimage import io, transform

from stanford40_dataset import Stanford40Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, models, transforms
import torchvision

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode
            
            model.train()
    
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.argmax(dim=1)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_lst,maps = model(inputs)
                    outputs = outputs_lst[-1]
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == "train":
                train_loss_hist.append(epoch_loss)
            else:
                val_loss_hist.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_hist, val_loss_hist

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=8):
    was_training = model.training
    model.train()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs_lst, maps = model(inputs)
            outputs = outputs_lst[-1]
            _, preds = torch.max(outputs, 1)
            
            I_train = utils.make_grid(inputs[:num_images,:,:,:], nrow=2, normalize=False, scale_each=True)
                
            img1, a1, attn1 = visualize_attn_softmax(I_train, c1, up_factor=8, nrow=2)
            img2, a2, attn2 = visualize_attn_softmax(I_train, c2, up_factor=4, nrow=2)
            img3, a3, attn3 = visualize_attn_softmax(I_train, c3, up_factor=4, nrow=2)

            imshow(attn1)
            imshow(attn2)
            imshow(attn3)

            for j in range(inputs.size()[0]):
                
                images_so_far += 1
                index = labels[j].argmax(dim=0)
                print('predicted: {}, actual: {}'.format(class_names[preds[j]], class_names[index]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.4*img + 0.6*attn
    return torch.from_numpy(img).permute(2,0,1), torch.from_numpy(attn).permute(2,0,1), torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.4 * img + 0.6 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def plot_loss(train_loss_hist, val_loss_hist):
    plt.figure()
    plt.plot(range(EPOCHS), train_loss_hist, label='Training Loss')
    plt.plot(range(EPOCHS), val_loss_hist, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()

class_names = ['applauding',
'blowing_bubbles',
'brushing_teeth',
'cleaning_the_floor',
'climbing',
'cooking',
'cutting_trees',
'cutting_vegetables',
'drinking',
'feeding_a_horse',
'fishing',
'fixing_a_bike',
'fixing_a_car', 
'gardening',
'holding_an_umbrella',
'jumping',
'looking_through_a_microscope',
'looking_through_a_telescope',
'playing_guitar',
'playing_violin',
'pouring_liquid',
'pushing_a_cart',
'reading',
'phoning',
'riding_a_bike',
'riding_a_horse',
'rowing_a_boat',
'running',
'shooting_an_arrow',
'smoking',
'taking_photos',
'texting_message',
'throwing_frisby',
'using_a_computer',
'walking_the_dog',
'washing_dishes',
'watching_TV',
'waving_hands',
'writing_on_a_board',
'writing_on_a_book']