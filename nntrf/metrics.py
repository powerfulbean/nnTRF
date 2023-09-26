# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:55:06 2020

@author: Jin Dou
"""
import numpy as np
# from scipy import stats as spStats

def Pearsonr(x,y):
    nObs = len(x)
    sumX = np.sum(x,0)
    sumY = np.sum(y,0)
    sdXY = np.sqrt((np.sum(x**2,0) - (sumX**2/nObs)) * (np.sum(y ** 2, 0) - (sumY ** 2)/nObs))
    
    r = (np.sum(x*y,0) - (sumX * sumY)/nObs) / sdXY
    return r

#def Pearsonr(x,y):
##    x = np.squeeze(x)
##    y = np.squeeze(y)
##    print(x.shape)
##    print(y.shape)
#    
#    out = spStats.pearsonr(x,y)
##    print('correlation',out)
#    return out

def BatchPearsonr(pred,y):
    result = list()
    for i in range(len(pred)):
        out1 = Pearsonr(pred[i],y[i])
#        print(out1)
        result.append(out1)
    return np.mean(result,0)