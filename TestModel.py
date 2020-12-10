# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:14:06 2020

@author: Jin Dou
"""

import torch
import torch.nn as nn

class CLinear(nn.Module):
    
    def __init__(self,inDim,outDim):
        super().__init__()
        self.oDense = nn.Sequential(
                nn.Linear(inDim,outDim),
                )
        
    def forward(self,x):
        #input shape ( batch, num_seq,input_size)     
        xDenseOut = self.oDense(x)
        return xDenseOut
    