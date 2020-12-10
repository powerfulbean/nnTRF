# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:24:45 2020

@author: Jin Dou
"""
import sys
sys.path.append('..')
sys.path.append('../mTRFpy/')
import torch
#from mTRF import Operation as trfOp

#class CLagTimeDataset(torch.utils.data.Dataset):
#    
#    def __init__(self, *tensors,tmin,tmax,fs):
#        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#        lags = trfOp.msec2Idxs([tmin,tmax],fs)
#        self.tmin = tmin
#        self.tmax = tmax
#        x = tensors[0].numpy()
#        x = trfOp.genLagMat(x,lags,Zeropad=True,bias=False)
#        x = torch.FloatTensor(x)
#        y = tensors[1]
#        self.tensors = tuple()
#        Temp = [x,y]
#        for tensor in Temp:
#            self.tensors += (tensor.cuda(),)
#
#    def __getitem__(self, index):
#        return tuple(tensor[index] for idx,tensor in enumerate(self.tensors))
#
#    def __len__(self):
#        return self.tensors[0].size(0)
    

class CEEGPredictDataset(torch.utils.data.Dataset):
    
    def __init__(self, *tensors):
#        print(tensors)
        self.tensors = tuple()
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        for tensor in tensors:
            self.tensors += (tensor.cuda(),)

    def __getitem__(self, index):
        return tuple(tensor[index] for idx,tensor in enumerate(self.tensors))

    def __len__(self):
        return self.tensors[0].size(0)
    
class CSeriesDataset(torch.utils.data.Dataset):
    
    def __init__(self, tensorsList):
        #*tensors: *[list of x ,list of y]
#        print(tensors)
        self.data = tensorsList

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)