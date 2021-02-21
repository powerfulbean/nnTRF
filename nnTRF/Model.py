# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:11:27 2020

@author: Jin Dou
"""


import torch
import numpy as np

def msec2Idxs(msecRange,fs):
    '''
    convert a millisecond range to a list of sample indexes
    
    the left and right ranges will both be included
    '''
    assert len(msecRange) == 2
    
    tmin = msecRange[0]/1e3
    tmax = msecRange[1]/1e3
    return list(range(int(np.floor(tmin*fs)),int(np.ceil(tmax*fs)) + 1))

def Idxs2msec(lags,fs):
    '''
    convert a list of sample indexes to a millisecond range
    
    the left and right ranges will both be included
    '''
    temp = np.array(lags)
    return list(temp/fs * 1e3)

def genLagMat(x,lags,Zeropad:bool = True,bias =True): #
    '''
    build the lag matrix based on input.
    x: input matrix
    lags: a list (or list like supporting len() method) of integers, 
         each of them should indicate the time lag in samples.
    
    see also 'lagGen' in mTRF-Toolbox https://github.com/mickcrosse/mTRF-Toolbox
    '''
    nLags = len(lags)
    
    nSamples = x.shape[0]
    nVar = x.shape[1]
    lagMatrix = np.zeros((nSamples,nVar*nLags))
    
    for idx,lag in enumerate(lags):
        colSlice = slice(idx * nVar,(idx + 1) * nVar)
        if lag < 0:
            lagMatrix[0:nSamples + lag,colSlice] = x[-lag:,:]
        elif lag > 0:
            lagMatrix[lag:nSamples,colSlice] = x[0:nSamples-lag,:]
        else:
            lagMatrix[:,colSlice] = x
    
    if bias:
        lagMatrix = np.concatenate([np.ones((lagMatrix.shape[0],1)),lagMatrix],1);

#    print(lagMatrix.shape)    
    
    return lagMatrix

class CTRF(torch.nn.Module):
    # the shape of the input for the forward should be the (nBatch,nTimeSteps,nChannels) 
    
    def __init__(self,inDim,outDim,tmin_ms,tmax_ms,fs):
        super().__init__()
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.fs = fs
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        self.realInDim = len(self.lagIdxs) * inDim
        self.oDense = torch.nn.Linear(self.realInDim,outDim)
        
    def timeLagging(self,tensor):
        x = tensor
        nBatch = x.shape[0]
        batchList = []
        for batchId in range(nBatch):
            batch = x[batchId:batchId+1]
            lagDataList = []
            for idx,lag in enumerate(self.lagIdxs):
                # we assume the last second dimension indicates time steps
                if lag < 0:
                    temp = torch.nn.functional.pad(batch,((0,0,0,-lag)))
#                    lagDataList.append(temp[:,-lag:,:])
                    lagDataList.append((temp.T)[:,-lag:].T)
                elif lag > 0:
                    temp = torch.nn.functional.pad(batch,((0,0,lag,0)))
#                    lagDataList.append(temp[:,0:-lag,:])
                    lagDataList.append((temp.T)[:,0:-lag].T)
                else:
                    lagDataList.append(batch)
            batchList.append(torch.cat(lagDataList,-1))
        x3 = torch.cat(batchList,0)
        return x3
    
    def forward(self,x):
        x = self.timeLagging(x)
        return self.oDense(x)
    
    @property
    def weights(self):
        return self.state_dict()['oDense.weight'].cpu().detach().numpy()
    
        