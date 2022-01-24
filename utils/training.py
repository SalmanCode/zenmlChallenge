#from utils.io import *
import torchvision.transforms as transforms

import torch, math
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

#from torchsummary import summary

from math import ceil
import os, glob
import pandas as pd
from torchvision.io import read_image
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange




def main():

    d_dict = {}
    for i, line in enumerate(open('./tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i 
        
    batch_size = 8
    transform = transforms.Normalize((122.4786, 114.2755, 101.3963),
                                 (70.4924, 68.5679, 71.8127))
    trainset = TrainTinyImageNetDataset(id=id_dict, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = TestTinyImageNetDataset(id=id_dict, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    print('check')
    for epoch in range(2):  # loop over the dataset multiple times
        t0 = time.time()
        epoch_accuracy = 0
        epoch_loss = 0
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        correct = 0
        total = 0
        correct_1 = 0
        correct_5 = 0
        c = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                #         outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                res = accuracy(outputs, labels)
                correct_1 += res[0][0].float()
                correct_5 += res[1][0].float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c += 1

        print("Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - Top 1: {correct_1 / c:.2f} - Top 5: {correct_5 / c:.2f} - Time: {time.time() - t0:.2f}\n")
        
        top1.append(correct_1 / c)
        top5.append(correct_5 / c)
        if float(correct_1 / c) >= float(max(top1)):
            PATH = 'CCN.pth'
            torch.save(model.state_dict(), PATH)
            print(1)
    print('Finished Training')
    
    
if __name__ == '__main__':
    main()