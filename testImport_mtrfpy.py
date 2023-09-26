# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:05:23 2022

@author:Jin Dou
"""

from StellarInfra import siIO
from matplotlib import pyplot as plt
from nntrf.Metrics import Pearsonr
from nntrf.Model import CCNNTRF, CTRF
from StretchedTRF.Model import CTransformTRFTwoStage,CMixedRF
from ResearchCode.nnTRFPlus.dyTimeStimuli.CustomWare import prepareDatasets
from mTRFpy import Model as mtrfModel
from mTRFpy import DataStruct as mtrfDS
import torch
import numpy as np
from StimRespFlow.DataProcessing.DeepLearning.Factory import CTorchDataset
seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda')
# path = r"D:\OneDrive\UR\Lab\Project\TransformerResearch\result\CompareGPT2Size\datasetName='LalorLab.NaturalSpeech'\gpt2Size='GPT2Small'_nSplits=10_stimFilterKeys=['onset', 'env', 'vector', 'tIntvl']_timeLag=[0, 800]\0\savedModel_9.mtrf"
path = r'.\testModel.mtrf'
# mTRFparam = siIO.loadObject(path)
# plt.figure()
# plt.plot(mTRFparam['w'][0])
# plt.plot(mTRFparam['w'][1])
# plt.plot(mTRFparam['w'][2])

ds = prepareDatasets(['NS'],vectorType = 'Surprisal',addEnvelope = True)
ds.stimFilterKeys = ['onset','env','vector','tIntvl']
ds = ds.nestedKFold(9,10)['train']
oTorchDS = CTorchDataset(ds)
dldr = torch.utils.data.dataloader.DataLoader(oTorchDS)
oNumpyDS = mtrfDS.buildListFromSRFDataset(ds,zscore = False)

oMTRF = mtrfModel.CTRF().load(path)
# oMTRF = mtrfModel.CTRF()
stim = oNumpyDS[0]
resp = oNumpyDS[1]
# oMTRF.train(stim, resp, 1, 64, -100, 700, 100)
mTRFparam = oMTRF.save()
plt.figure()
plt.plot(mTRFparam['w'][0])
plt.plot(mTRFparam['w'][1])
plt.plot(mTRFparam['w'][2])
linW = mTRFparam['w']
linB = mTRFparam['b']
print(linW,linB)



oTRF2 = CCNNTRF(3, 128, -100, 700, 64)
oTRF2.loadFromMTRFpy(mTRFparam['w'], mTRFparam['b'],device)
oTRF2.eval()

oTRF3 = CTRF(3, 128,-100, 700, 64)
oTRF3.loadFromMTRFpy(mTRFparam['w'], mTRFparam['b'],device)
oTRF3.eval()
plt.figure()
plt.plot(oTRF3.w[0])
plt.plot(oTRF3.w[1])
plt.plot(oTRF3.w[2])

oTRFs = torch.nn.ModuleDict()
oTRF = CCNNTRF(2, 128, -100, 700, 64)
oTRF.loadFromMTRFpy(mTRFparam['w'][0:2], mTRFparam['b']/2,device)
oTRFs['NS'] = oTRF
oTRFs.eval()
oTRF = oTRF.to(device)
oNonLinTRF = CTransformTRFTwoStage('CLSTMSeqContexter',1,128,-100,700,64,device,['NS'],nNonLinState = 8,ifLite= True,constrain = 'LimitedSource')
oNonLinTRF.loadFromMTRFpy(mTRFparam['w'][2:], mTRFparam['b']/2,'NS')
oMixedTRF = CMixedRF(device, oTRFs, oNonLinTRF,ifZscore = False)
oMixedTRF = oMixedTRF.to(device)
oMixedTRF.eval()
state_dict = torch.load('testNNModel.th')
oMixedTRF.load_state_dict(state_dict)

plt.figure()
plt.plot(oTRF.w[0])
plt.plot(oTRF.w[1])
plt.plot(oNonLinTRF.LinearKernels['NS'].weight[0].cpu().detach().numpy())

mTRFpyInput = oNumpyDS[0][0]
mTRFpyInput = mTRFpyInput[0:11289]
nnTRFInput = next(iter(dldr))
mTRFpyInput2 = torch.load('testMTRFModel.input')
nnTRFInput = torch.load('testNNModel.input')
# nnTRFInput[0]['env'] = nnTRFInput[0]['env'][:,:,:11289]
# nnTRFInput[0]['onset'] = nnTRFInput[0]['onset'][:,:,:11289]
# nnTRFInput[0]['tIntvl'] = nnTRFInput[0]['tIntvl'][:,:,:583]
# nnTRFInput[0]['vector'] = nnTRFInput[0]['vector'][:,:,:583]

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

print(predNNTRF[-1])

assert np.allclose(predNNTRF2,predTRFpy,rtol=1e-05, atol=1e-07)
assert np.allclose(predNNTRF3,predTRFpy,rtol=1e-05, atol=1e-07)
assert np.allclose(predNNTRF,predTRFpy,rtol=1e-05, atol=1e-07)
assert np.allclose(predNNTRF2,predNNTRF3,rtol=1e-05, atol=1e-07)


# _,r,err = oMTRF.predict(mTRFpyInput,oNumpyDS[1][0])
