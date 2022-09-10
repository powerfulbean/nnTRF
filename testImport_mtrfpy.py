# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:05:23 2022

@author:Jin Dou
"""

from StellarInfra import siIO
from matplotlib import pyplot as plt
from nnTRF.Metrics import Pearsonr
from nnTRF.Model import CCNNTRF, CTRF
from StretchedTRF.Model import CTransformTRFTwoStage,CMixedRF
from ResearchCode.nnTRFPlus.dyTimeStimuli.CustomWare import prepareDatasets
from mTRFpy import Model as mtrfModel
from mTRFpy import DataStruct as mtrfDS
import torch
import numpy as np
from StimRespFlow.DataProcessing.DeepLearning.Factory import CTorchDataset
device = torch.device('cuda')
path = r"D:\OneDrive\UR\Lab\Project\TransformerResearch\result\CompareGPT2Size\datasetName='LalorLab.NaturalSpeech'\gpt2Size='GPT2Small'_nSplits=10_stimFilterKeys=['onset', 'env', 'vector', 'tIntvl']_timeLag=[0, 800]\0\savedModel_9.mtrf"
mTRFparam = siIO.loadObject(path)
plt.figure()
plt.plot(mTRFparam['w'][0])
plt.plot(mTRFparam['w'][1])
plt.plot(mTRFparam['w'][2])

oTRFs = torch.nn.ModuleDict()
oTRFs.eval()
oTRF = CCNNTRF(2, 128, 0, 800, 64)
oTRF.loadFromMTRFpy(mTRFparam['w'][0:2], mTRFparam['b'][0:2]/2,device)
oTRFs['NS'] = oTRF

oTRF2 = CCNNTRF(3, 128, 0, 800, 64)
oTRF2.loadFromMTRFpy(mTRFparam['w'], mTRFparam['b'],device)
oTRF2.eval()

oTRF3 = CTRF(3, 128, 0, 800, 64)
oTRF3.loadFromMTRFpy(mTRFparam['w'], mTRFparam['b'],device)
oTRF3.eval()
plt.figure()
plt.plot(oTRF3.w[0])
plt.plot(oTRF3.w[1])
plt.plot(oTRF3.w[2])

oTRF = oTRF.to(device)
oNonLinTRF = CTransformTRFTwoStage('CSelfAttnSeqContexter',1,128,0,800,64,device,['NS'])
oNonLinTRF.loadFromMTRFpy(mTRFparam['w'][2:], mTRFparam['b'].squeeze()/2,'NS')
oMixedTRF = CMixedRF(device, oTRFs, oNonLinTRF,ifZscore = True)
oMixedTRF = oMixedTRF.to(device)
oMixedTRF.eval()

plt.figure()
plt.plot(oTRF.w[0])
plt.plot(oTRF.w[1])
plt.plot(oNonLinTRF.LinearKernels['NS'].weight[0].cpu().detach().numpy())


ds = prepareDatasets(['NS'],vectorType = 'Surprisal',addEnvelope = True)
ds.stimFilterKeys = ['onset','env','vector','tIntvl']
oMTRF = mtrfModel.CTRF().load(path)
oTorchDS = CTorchDataset(ds)
dldr = torch.utils.data.dataloader.DataLoader(oTorchDS)
oNumpyDS = mtrfDS.buildListFromSRFDataset(ds)

mTRFpyInput = oNumpyDS[0][0]
nnTRFInput = next(iter(dldr))

plt.figure()
plt.plot(mTRFpyInput[:,2])
plt.plot(nnTRFInput[0]['env'].numpy().squeeze())

plt.figure()
plt.plot(mTRFpyInput[:,1])
plt.plot(nnTRFInput[0]['onset'].numpy().squeeze())

predTRFpy = oMTRF.predict(mTRFpyInput)[0]
predNNTRF = oMixedTRF(*oMixedTRF.parseBatch(nnTRFInput))[0].detach().cpu().numpy()[0].T
predNNTRF2 = oTRF2(torch.FloatTensor(mTRFpyInput.T).to(device).unsqueeze(0))[0].detach().cpu().numpy().T
predNNTRF3 = oTRF3(torch.FloatTensor(mTRFpyInput).to(device).unsqueeze(0))[0].detach().cpu().numpy()

assert np.allclose(predNNTRF2,predTRFpy,rtol=1e-05, atol=1e-07)