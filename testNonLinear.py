import numpy as np
import torch
from mtrf.model import TRF, load_sample_data
from nntrf.models import ASTRF, FuncTRFsGen
device = torch.device('cuda')

def testCanRun():
    model = ASTRF(1, 128, 0, 700, 64, device = device)
    x = torch.rand((1,1,10))
    timeinfo = torch.tensor([
        [[1,3,5, 7,20, 40,45,60,75,90],
        [1,3,5, 7,20, 40,45,60,75,90]]]).float()
    x = x.to(device)
    timeinfo = timeinfo.to(device)

    output1 = model(x, timeinfo)
    print(output1.shape)

    trfsGen = FuncTRFsGen(1, 128, 0, 700, 64, device = device)

    model.setTRFsGen(trfsGen)
    model.ifEnableUserTRFGen = True
    output2 = model(x, timeinfo)
    print(output2.shape)
    assert not torch.equal(output1, output2)

    model.ifEnableUserTRFGen = False
    output3 = model(x, timeinfo)
    print(output3.shape)
    assert torch.equal(output1, output3)

def testLTIWeight():
    stimulus, response, fs = load_sample_data(n_segments=9)
    trf = TRF(direction=1)
    trf.train(stimulus, response, fs, 0, 0.7, 100)
    model = ASTRF(16, 128, 0, 700, fs, device = device)
    model.loadLTIWeights(trf.weights, trf.bias)
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)
    x = torch.stack([torch.tensor(i.T) for i in stimulus], dim = 0).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]
    timeinfo = [None for i in range(nBatch)]
    predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()

    # print(predNNTRF, predMTRF)
    # print(predNNTRF.shape, predMTRF.shape)
    assert np.allclose(predNNTRF, predMTRF, atol = 1e-6)

stimulus, response, fs = load_sample_data(n_segments=9)
model = ASTRF(16, 128, 0, 700, fs, device = device)
trfsGen = FuncTRFsGen(16, 128, 0, 700, fs, device = device)
extLagMin, extLagMax = trfsGen.extendedTimeLagRange
trf = TRF(direction=1)
trf.train(stimulus, response, fs, extLagMin/1000, extLagMax/1000, 100)
trfsGen.fitFuncTRF(trf.weights)
fig = trfsGen.funcTRF.visResult()
fig.savefig('funcTRF.png')
model.setTRFsGen(trfsGen)
model.ifEnableUserTRFGen = True
model = model.eval()

predMTRF = trf.predict(stimulus)
predMTRF = np.stack(predMTRF, axis = 0)
x = torch.stack([torch.tensor(i.T) for i in stimulus], dim = 0).to(device).float()
nBatch = x.shape[0]
nSeq = x.shape[2]
timeinfo = [None for i in range(nBatch)]
with torch.no_grad():
    predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
# print(predNNTRF, predMTRF)

from scipy.stats import pearsonr
nBatch, nSeq, nChan = predNNTRF.shape
rs = []
for b in range(nBatch):
    for c in range(nChan):
        r = pearsonr(predNNTRF[b,:,c], predMTRF[b, :, c])[0]
        rs.append(r)
print(np.mean(rs))

with torch.no_grad():
    model.trfsGen.funcTRF.saveMem = True
    predNNTRF2 = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
print(predNNTRF, predNNTRF2)
assert np.allclose(predNNTRF, predNNTRF2, atol = 1e-6)