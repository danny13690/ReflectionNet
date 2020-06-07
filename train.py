from __future__ import print_function, division

import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from skimage import io, transform

from data_processing.stanford40_dataset import Stanford40Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision import datasets, models, transforms
import torchvision

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                    outputs,c1,c2,c3 = model(inputs)
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
    return model


def get_data():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train = Stanford40Dataset("../data/Stanford40", "train", transform=data_transforms['train'])
    train_loader = DataLoader(train, batch_size=4, num_workers=4, shuffle=True)

    test = Stanford40Dataset("../data/Stanford40", "test", transform=data_transforms['test'])
    val_set, test_set = torch.utils.data.random_split(test, [2766, 2766])
    val_loader = DataLoader(val_set, batch_size=4, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4, num_workers=4)

    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataset_sizes = {"train": len(train), "val": len(val_set), "test": len(test_set)}

    return dataloaders, dataset_sizes

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

def visualize_model(model, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, c1,c2,c3 = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

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