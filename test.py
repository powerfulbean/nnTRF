# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 22:35:50 2020

@author: Jin Dou
"""
import torch
import numpy as np
from nnTRF.Model import CTRF,CCNNTRF
# lagIdxs = [-1,0,1,2]
# data1 = np.expand_dims(np.array([[1,2,3,4,5],[-1,-2,-3,-4,-5]]).T,0)
# data2 = np.expand_dims(np.array([[2,3,4,5,6],[-2,-3,-4,-5,-6]]).T,0)

# oriData = np.concatenate([data1, data2])
# x = torch.Tensor(oriData)
# x.requires_grad = True
# #last dimension will be channel
# #nBatch = x.shape[0]
# #nLags = len(lagIdxs)
# #batchList = []
# #for batchId in range(nBatch):
# #    batch = x[batchId:batchId+1]
# #    lagDataList = []
# #    for idx,lag in enumerate(lagIdxs):
# #        lagData = None
# #        if lag < 0:
# #            temp = torch.nn.functional.pad(batch,((0,0,0,-lag)))
# #            lagDataList.append(temp[:,-lag:,:])
# #        elif lag > 0:
# #            temp = torch.nn.functional.pad(batch,((0,0,lag,0)))
# #            lagDataList.append(temp[:,0:-lag,:])
# #        else:
# #            lagDataList.append(batch)
# #    batchList.append(torch.cat(lagDataList,-1))
# #x3 = torch.cat(batchList,0)
# oModel = CTRF(2,3,-1,2,1000)
# x3 = oModel.timeLagging(x)
# #oDense = torch.nn.Linear(2,1)     
    

# #x1 = x.repeat(1, 1,5)
# #x2 = x1[:,0:4,:]
# #x3 = torch.cat([x2,x2],1)
# #y = oDense(x)
# y = torch.sum(x3**3)
# y.backward()
# print(x.grad is None)


oModel2 = CCNNTRF(769, 128, 0, 700, 64)
path = r"D:\OneDrive\Code\Rotation3\semanticEncoding\result\xlnet_finetune4_onset_read_-0.52021_04_29T00_42_48.299622\savedModel_feedForward_best.pt"
oModel2.load(path)

from matplotlib import pyplot as plt
