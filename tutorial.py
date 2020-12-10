# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 10:15:43 2020

@author: Jin Dou

"""
import sys
sys.path.append('../StellarInfra/')
sys.path.append('../mTRFpy/')
sys.path.append('..')
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np

from StellarInfra import DirManage as siDM
from StellarInfra import IO as siIO
from StellarInfra.Logger import CLog
from StellarBrainwav.DataStruct.DataSet import CDataRecord
from StellarBrainwav.Helper.DataObjectTransform import CDataRecordToTensors
from StellarBrainwav.DataProcessing.DeepLearning import CPytorch

from mTRFpy import Operations as trfOp

from TestModel import CLinear
from TorchData import CEEGPredictDataset,CSeriesDataset
from nnTRF.Model import CTRF


#def getWeights(model):
#    temp = dict()
#    for name,param in model.named_parameters():
#        temp[name] = param  
#    weights = temp['oDense.0.weight'].cpu().detach().numpy()
#    weights = weights.reshape(128,46,2)
#    return weights

def plotWeights(weights,times,Dir = -1):
        
    weights = weights.reshape(128,46,2)
    
    fig1 = plt.figure()
    plt.plot(times,weights[:,:,0].T)
    fig2 = plt.figure()
    plt.plot(times,weights[:,:,1].T)
    fig3 = plt.figure()
    import numpy as np
    temp = np.mean(weights,0)
    plt.plot(times,temp[:,1])
    return fig1,fig2,fig3

oDir = siDM.CDirectoryConfig(['Root','Results'],'.\Dataset.conf')
FolderName = 'SemanticDisimilarityTRF'
tarFolder = oDir.Results + FolderName + '/'
oLog = CLog(tarFolder,'SemanticDisimilarityTRF')
oLog('test with new nnTRF.Model module')




files = siDM.getFileList(oDir.Root,'.mat')
matData = siIO.loadMatFile(files[2])

resp = matData['respData']
stim = matData['stimData']
trialStartIdx = matData['trialStartIdx']

#respTrain = resp[0:trialStartIdx[-1],:]
#stimTrain = resp[0:trialStartIdx[-1],:]
#respTest = stim[trialStartIdx[-1]:,:]
#stimTest = stim[trialStartIdx[-1]:,:]
#del resp
#del stim
oDataRecord = CDataRecord(stim.T,resp.T,['onset','semantic'],64)

tmin = 0
tmax = 700
fs = 64
#argsTrain = {'DatasetArgs':{'tmin':tmin,'tmax':tmax,'fs':fs},
#            'DataLoaderArgs':{'shuffle':False,'batch_size':1000},
#            }
#argsTest = {'DatasetArgs':{'tmin':tmin,'tmax':tmax,'fs':fs},
#            'DataLoaderArgs':{'shuffle':False,'batch_size':1000},
#            }

oDataTrans = CDataRecordToTensors()
oTensorPairList = []
trialStartIdx = list(np.squeeze(trialStartIdx))
for s,e in zip(trialStartIdx,trialStartIdx[1:] + [len(stim)]):
    stimSeg = stim[s:e]
    respSeg = resp[s:e]
    oTensorPairList.append(oDataTrans(CDataRecord(stimSeg.T,respSeg.T,['onset','semantic'],64)))
    
dataset = CSeriesDataset(oTensorPairList)

#sys.exit(0)

#oTorch = CPytorch()

#tensorsTrain = oDataTrans(oDataRecord)
#dataset = CEEGPredictDataset(*tensorsTrain)
dataloaderTrain = torch.utils.data.DataLoader(dataset,batch_size = 1, shuffle=False)
#exampleTensors = dataloaderTrain.dataset.tensors
#inDim,outDim = exampleTensors[0].shape[1],exampleTensors[1].shape[1]
#model = CLinear(inDim,outDim)
#model 
model = CTRF(2,128,tmin,tmax,fs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#optimizier = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizier = torch.optim.AdamW(model.parameters(), lr=1e-2,weight_decay= 0.01)
criterion = torch.nn.MSELoss()

nEpoch = 20
for epoch in range(nEpoch):
    iterator = iter(dataloaderTrain) 
    smoothRatio = 0.99
    with trange(len(dataloaderTrain)) as t:
        for idx in t:
            sample = next(iterator)
            x,y = sample[0].to(device),sample[1].to(device)
            model.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            optimizier.step()
            t.set_description(f"epoc : {epoch}, loss {loss}")
    if (epoch + 1)%5 == 0:
        [f1,f2,f3] = plotWeights(model.weights,model.lagTimes)
        f1.savefig(tarFolder + 'epoch_' + str(epoch) + '_onset' + '.png')
        f2.savefig(tarFolder + 'epoch_' + str(epoch) + '_semantic' + '.png')
        f3.savefig(tarFolder + 'epoch_' + str(epoch) + '_semanticMean' + '.png')
        plt.close(f1)
        plt.close(f2)
        plt.close(f3)  

corr = trfOp.pearsonr(y.detach().cpu().numpy(),pred.detach().cpu().numpy())

    

    
    




