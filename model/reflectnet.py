import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .blocks import *

class ReflectNet(nn.Module):
    def __init__(self, num_passes):
        super().__init__()

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features

        conv1, bn1, relu1, maxpool1, seq1, seq2, seq3, seq4, avgpool, fc = tuple(model_ft.children())
        
        self.conv1 = conv1
        self.batchnorm1 = bn1
        self.relu1 = relu1
        self.maxpool1 = maxpool1
        self.seq1 = seq1
        self.seq2 = seq2
        self.seq3 = seq3
        self.seq4 = seq4

        self.averagepool = avgpool

        self.lstm = nn.LSTM(input_size=num_ftrs, hidden_size=100)
        self.query = QueryBlock(100, 1024, 512)
        self.attn = LinearAttentionBlock(512)

        self.num_passes = num_passes

        #self.fc = nn.Linear(num_ftrs,40)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        out0 = self.maxpool1(x)
        
        out1 = self.seq1(out0)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        out4 = self.seq4(out3)
        
        input = self.averagepool(out4)

        results = []
        maps = []
        hidden = (torch.randn(1, 1, 100),torch.randn(1, 1, 100))

        for i in range(self.num_passes):
            out, hidden = self.lstm(input.view(1,1,-1), hidden)
            query = self.query(hidden)
            c, g = self.attn(out4, query)
            input = self.averagepool(g)
            results.append(out)
            maps.append(c)
        
        return [results, maps]