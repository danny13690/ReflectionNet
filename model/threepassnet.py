import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import *

class ThreePassNet(nn.Module):
    def __init__(self):
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
        self.fc = nn.Linear(num_ftrs,40)
        

        self.attn1 = GridAttentionBlock(128, 512, 128, 4, normalize_attn=True)
        self.attn2 = GridAttentionBlock(64, 256, 64, 4, normalize_attn=True)
        self.attn3 = GridAttentionBlock(64, 128, 64, 2, normalize_attn=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        out0 = self.maxpool1(x)
        
        out1 = self.seq1(out0)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        out4 = self.seq4(out3)
        
        c1, g1 = self.attn1(out2,out4)
        
        newout3 = self.seq3(g1)
        
        c2, g2 = self.attn2(out1,newout3)
        
        newout2 = self.seq2(g2)
        
        c3, g3 = self.attn3(out0,out2)
        
        newout1 = self.seq1(g3)
        
        finalout = self.seq2(newout1)
        finalout = self.seq3(finalout)
        finalout = self.seq4(finalout)
        
        result = self.averagepool(finalout)
        result = result.view(-1, 512)
        result = self.fc(result)
        
        return [result, c1, c2, c3]