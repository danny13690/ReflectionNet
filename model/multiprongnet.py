import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .blocks import *

class MultiProngNet(nn.Module):
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

        self.query = QueryBlock(512, 1024, 64+128+256)
        self.attn1 = LinearAttentionBlock(64)
        self.attn2 = LinearAttentionBlock(128)
        self.attn3 = LinearAttentionBlock(256)

        self.fc = nn.Linear(3*512,40)
        
    def forward(self, x):
        B = x.shape[0]

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        out0 = self.maxpool1(x)
        
        out1 = self.seq1(out0)
        out2 = self.seq2(out1)
        out3 = self.seq3(out2)
        out4 = self.seq4(out3)

        final0 = self.averagepool(out4)
        query = self.query(final0.view(B,-1))

        c1, g1 = self.attn1(out1,query[:,:64].view(B,64,1,1))
        c2, g2 = self.attn2(out2,query[:,64:64+128].view(B,128,1,1))
        c3, g3 = self.attn3(out3,query[:,64+128:].view(B,256,1,1))

        final1 = self.seq2(g1)
        final1 = self.seq3(final1)
        final1 = self.seq4(final1)
        final1 = self.averagepool(final1)

        final2 = self.seq3(g2)
        final2 = self.seq4(final2)
        final2 = self.averagepool(final2)

        final3 = self.seq4(g3)
        final3 = self.averagepool(final3)

        output = torch.cat((final1,final2,final3),1).view(B,-1)
        output = self.fc(output)
        
        return (output, [c1,c2,c3])
