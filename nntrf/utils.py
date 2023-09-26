# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:08:45 2020

@author: Jin Dou
"""

def TensorsToNumpy(*tensors):
    out = tuple()
    for tensor in tensors:
        out += (tensor.cpu().detach().numpy(),)
    return out