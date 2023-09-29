import numpy as np
import torch
from mtrf.model import TRF, load_sample_data
from nntrf.models import ASTRF, FuncTRFsGen
device = torch.device('cuda')

def testCanRun():
    trf = TRF()
    model = ASTRF(1, 128, 0, 700, 64, device = device)
    w,b = model.exportLTIWeights()
    trf.weights = w
    trf.bias = b
    trf.times = np.array(model.lagTimes) / 1000
    trf.fs = model.fs
    x = [torch.rand((1, 10)).to(device), torch.rand((1, 8)).to(device)]
    timeinfo = [
        torch.tensor(
            [
                [1,3,5,7,20,40,45,60,75,90],
                [1,3,5,7,20,40,45,60,75,90]
            ]
        ).float().to(device),
        torch.tensor(
            [
                [1,3,5,7,45,60,75,90],
                [1,3,5,7,45,60,75,90]
            ]
        ).float().to(device)
    ]

    trfInput = []
    for idx, t in enumerate(timeinfo):
        t1 = t.cpu().numpy()
        nLen = np.ceil(t1[0][-1] * trf.fs).astype(int) + model.nWin
        vIdx = np.round(t1[0] * trf.fs).astype(int)
        vec = np.zeros((nLen, 1))
        vec[vIdx,:] = x[idx].cpu().numpy().T
        trfInput.append(vec)
    trfOutput = trf.predict(trfInput)
    output1 = model(x, timeinfo)
    output11 = output1.cpu().detach().numpy()
    for idx, out in enumerate(trfOutput):
        t_nLen = out.shape[0]
        print(np.allclose(out, output11[idx].T[:t_nLen]))
        print(np.allclose(out, output11[idx].T[:t_nLen], atol = 1e-11))
        assert np.allclose(out, output11[idx].T[:t_nLen])

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
    stimulus = stimulus[:3]
    response = response[:3]
    trf = TRF(direction=1)
    trf.train(stimulus, response, fs, 0, 0.7, 100)
    model = ASTRF(16, 128, 0, 700, fs, device = device)
    model.loadLTIWeights(trf.weights, trf.bias)
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)
    x = torch.stack([torch.tensor(i.T) for i in stimulus], dim = 0).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]
    model = model.eval()
    timeinfo = [None for i in range(nBatch)]
    with torch.no_grad():
        predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()

    # print(predNNTRF, predMTRF)
    # print(predNNTRF.shape, predMTRF.shape)
    assert np.allclose(predNNTRF, predMTRF, atol = 1e-6)


def testFuncTRF():
    stimulus, response, fs = load_sample_data(n_segments=9)
    stimulus = stimulus[:3]
    response = response[:3]
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

    trf.times = trf.times[7:-7]
    trf.weights = trf.weights[:,7:-7,:]
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)
    x = torch.stack([torch.tensor(i.T) for i in stimulus], dim = 0).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]
    timeinfo = [None for i in range(nBatch)]
    with torch.no_grad():
        predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
    # print(predNNTRF, predMTRF)

    assert np.allclose(predNNTRF, predMTRF, atol = 1e-1)
    from scipy.stats import pearsonr
    nBatch, nSeq, nChan = predNNTRF.shape
    rs = []
    for b in range(nBatch):
        for c in range(nChan):
            r = pearsonr(predNNTRF[b,:,c], predMTRF[b, :, c])[0]
            rs.append(r)
    print(np.mean(rs))
    assert np.mean(rs) > 0.99
    with torch.no_grad():
        model.trfsGen.funcTRF.saveMem = True
        predNNTRF2 = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
    # print(predNNTRF, predNNTRF2)
    assert np.allclose(predNNTRF, predNNTRF2, atol = 1e-6)

testCanRun()
testLTIWeight()
testFuncTRF()