# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:14:06 2020

@author: Jin Dou
"""

import torch
import torch.nn as nn
import nnTRF.Model as nntrfModel

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
    
    
class CTanhshrink(nn.Module):
    
    def __init__(self,inDim,outDim,trfOutDim,tmin,tmax,fs):
        super().__init__()
        self.oTRF = nntrfModel.CTRF(inDim,trfOutDim,tmin,tmax,fs)
        self.oNonLinear = nn.Sequential(
#                nn.Linear(trfOutDim,64),
#                nn.LeakyReLU(),
                nn.Linear(trfOutDim,outDim),
                nn.Tanhshrink()
                )
        
    def forward(self,x):
        trfOut = self.oTRF(x)
        return self.oNonLinear(trfOut)