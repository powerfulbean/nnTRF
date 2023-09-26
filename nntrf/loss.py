# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:56:39 2020

@author: Jin Dou
"""
import torch
def pearsonCorrLoss(x,y):

    mx = torch.mean(x,1)
    my = torch.mean(y,1)
    x = x[:,:,0]
    y = y[:,:,0]
    xm, ym = x-mx, y-my
    r_num = torch.sum(xm * ym,1)
    r_den = torch.sqrt(torch.sum(xm*xm,1) * torch.sum(ym*ym,1))
#    print(torch.sum(r_num_sub)/r_den)
    if(torch.mean(r_num) == 0 and torch.mean(r_den) ==0):
        raise ValueError("gradient gone\n")
        r = None
    else:
        r = r_num / r_den
#    return 1 - r**2
        
#    print(r)
    r = torch.mean(r)
#    loss = torch.exp(r)
    loss = 1 - r

    return loss

