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
from sklearn.model_selection import KFold

from StellarInfra import DirManage as siDM
from StellarInfra import IO as siIO
from StellarInfra.Logger import CLog
from StellarBrainwav.DataStruct.DataSet import CDataRecord
from StellarBrainwav.Helper.DataObjectTransform import CDataRecordToTensors
from StellarBrainwav.DataProcessing.DeepLearning import CPytorch

from mTRFpy import Operations as trfOp

from TestModel import CLinear
from TorchData import CEEGPredictDataset,CSeriesDataset
from nntrf.Model import CTRF
from nntrf.Metrics import BatchPearsonr,Pearsonr
from nntrf.Utils import TensorsToNumpy

from TestModel import CTanhshrink


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
    print(temp.shape)
    plt.plot(times,temp[:,1])
    return fig1,fig2,fig3

def plotWeights2(weights,times,Dir = -1):
        
    weights = weights.reshape(128,64,1)
    
    fig1 = plt.figure()
    plt.plot(times,weights[:,:,0].T)
    fig3 = plt.figure()
    import numpy as np
    temp = np.mean(weights,0)
    print(temp.shape)
    plt.plot(times,temp[:,0])
    return fig1,fig3


def train(model,dataloader,optimizer,nEpoch,stage):
    model.train()
    for epoch in range(nEpoch):
        iterator = iter(dataloaderTrain) 
        lossList = []
        corrList = []
        with trange(len(dataloaderTrain)) as t:
            for idx in t:
                sample = next(iterator)
                x,y = sample[0].to(device),sample[1].to(device)
                model.zero_grad()
                pred = model(x)
                loss = criterion(pred,y)
                loss.backward()
                optimizer.step()
                tensors = TensorsToNumpy(pred,y)
                corr = np.mean(BatchPearsonr(*tensors))
                lossList.append(TensorsToNumpy(loss)[0])
                corrList.append(corr)
                t.set_description(f"epoc : {epoch}, loss {loss:.5f}, corr {corr:.5f}")
        
        oLog('epoch',epoch,'loss',np.average(lossList),'corr',np.average(corrList))
        if (epoch + 1) == nEpoch:
    #        [f1,f2,f3] = plotWeights(model.weights,model.lagTimes)
            if isinstance(model,CTRF):
                [f1,f2,f3] = plotWeights(model.weights,model.lagTimes)
            else:
                [f1,f2,f3] = plotWeights(model.oTRF.weights,model.oTRF.lagTimes)
            f1.savefig(tarFolder + '/' + stage + '_epoch_' + str(epoch) + '_onset' + '.png')
            f2.savefig(tarFolder +  '/' + stage + '_epoch_' + str(epoch) + '_semantic' + '.png')
            f3.savefig(tarFolder +  '/' + stage + '_epoch_' + str(epoch) + '_semanticMean' + '.png')
            plt.close(f1)
            plt.close(f2)
            plt.close(f3)  
    return np.average(lossList),np.average(corrList)
        
def test(model,dataloader):
    model.eval()
    iterator = iter(dataloaderTrain) 
    lossList = []
    corrList = []
    with trange(len(dataloaderTrain)) as t:
        for idx in t:
            sample = next(iterator)
            x,y = sample[0].to(device),sample[1].to(device)
            model.zero_grad()
            pred = model(x)
            loss = criterion(pred,y)
            loss.backward()
            tensors = TensorsToNumpy(pred,y)
            corr = np.mean(BatchPearsonr(*tensors))
            lossList.append(TensorsToNumpy(loss)[0])
            corrList.append(corr)
            t.set_description(f"loss {loss:.5f}, corr {corr:.5f}")
    
    oLog('testloss',np.average(lossList),'testcorr',np.average(corrList))
    return np.average(lossList),np.average(corrList)
        
oDir = siDM.CDirectoryConfig(['Root','Results'],'.\Dataset.conf')
FolderName = 'SemanticDisimilarityTRF_combineNonLinear'
tarFolder = oDir.Results + FolderName + '/'
siDM.checkFolder(tarFolder)
oLog = CLog(tarFolder,FolderName)
oLog('test combining trf and nonlinear, with cross validation')

oLog.ifPrint = False


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
    


#sys.exit(0)

#oTorch = CPytorch()

#tensorsTrain = oDataTrans(oDataRecord)
#dataset = CEEGPredictDataset(*tensorsTrain)

#exampleTensors = dataloaderTrain.dataset.tensors
#inDim,outDim = exampleTensors[0].shape[1],exampleTensors[1].shape[1]
#model = CLinear(inDim,outDim)
#model 
#model = CTRF(2,128,tmin,tmax,fs)
trainResult = []
testResult = []
for trainIdx ,testIdx in KFold(n_splits=15).split(oTensorPairList):
    oLog('testIdx',testIdx)
    oTrainList = [oTensorPairList[i] for i in trainIdx]
    oTestList = [oTensorPairList[i] for i in testIdx]
    datasetTrain = CSeriesDataset(oTrainList)
    datasetTest = CSeriesDataset(oTestList)
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,batch_size = 1, shuffle=False)
    dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size = 1, shuffle=False)
    
    model = CTanhshrink(2,128,128,tmin,tmax,fs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #optimizier = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizier1 = torch.optim.AdamW(model.oTRF.parameters(), lr=1e-2,weight_decay= 0.01)
    optimizier2 = torch.optim.AdamW(model.oNonLinear.parameters(), lr=1e-2,weight_decay= 0.01)
    optimizier = torch.optim.AdamW(model.parameters(), lr=1e-2,weight_decay= 0.01)
    criterion = torch.nn.MSELoss()
    train(model.oTRF,dataloaderTrain,optimizier1,20,'trf')
    loss,corr = train(model,dataloaderTrain,optimizier,50,'nonLinear')
    trainResult.append([loss,corr])
    loss,corr = test(model,dataloaderTest)
    testResult.append([loss,corr])
    



    

    
    




