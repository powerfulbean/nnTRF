import numpy as np
import torch
from mtrf.model import TRF, load_sample_data
from nntrf.models import ASTRF, FuncTRFsGen, WordTRFEmbedGen, WordTRFEmbedGenTokenizer
from nntrf.models import ASCNNTRF, CNNTRF, GaussianBasisTRF
device = torch.device('cpu')

def testASTRF():
    trf = TRF()
    model = ASTRF(1, 128, 0, 700, 64, device = device)
    w,b = model.get_linear_weights()
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
                [1,3,5,7,45,60,65,70],
                [1,3,5,7,45,60,65,70]
            ]
        ).float().to(device)
    ]

    #test the ASTRF and mTRF generate similar results
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

    #test ifEnableUserTRFGen works
    trfsGen = FuncTRFsGen(1, 128, 0, 700, 64, device = device)
    model.set_trfs_gen(trfsGen)
    model.if_enable_trfsGen = True
    output2 = model(x, timeinfo)
    print(output2.shape)
    assert not torch.equal(output1, output2)
    
    model.if_enable_trfsGen = False
    output3 = model(x, timeinfo)
    print(output3.shape)
    assert torch.equal(output1, output3)

def testASTRFLTI():
    stimulus, response, fs = load_sample_data(n_segments=9)
    stimulus = [s.mean(axis=1, keepdims=True) for s in stimulus]
    # stimulus = stimulus[:3]
    # response = response[:3]
    trf = TRF(direction=1)
    trf.train(stimulus, response, fs, 0, 0.7, 100)
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)
    x = torch.stack(
        [
            torch.tensor(i.T) for i in stimulus
        ], 
        dim = 0
    ).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]

    model = ASTRF(16, 128, 0, 700, fs, device = device)
    model.set_linear_weights(trf.weights, trf.bias)
    model = model.eval()
    timeinfo = [None for i in range(nBatch)]
    with torch.no_grad():
        predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
    # print(predNNTRF, predMTRF)
    # print(predNNTRF.shape, predMTRF.shape)
    assert np.allclose(predNNTRF, predMTRF, atol = 1e-6)

def testASCNNTRFLTI(tmin, tmax):
    print('testASCNNTRFLTI', tmin, tmax)
    stimulus, response, fs = load_sample_data(n_segments=1)
    trf = TRF(direction=1)
    trf.train(stimulus, response, fs, tmin / 1e3, tmax / 1e3, 100)
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)
    
    x = torch.stack(
        [
            torch.tensor(i.T) for i in stimulus
        ], 
        dim = 0
    ).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]

    base = CNNTRF(16, 128, tmin, tmax, fs)
    base.loadFromMTRFpy(trf.weights, trf.bias, device = device)
    base = base.eval()
    with torch.no_grad():
        predNNTRF2 = base(x).cpu().detach().permute(0,2,1).numpy()
    assert np.allclose(predNNTRF2, predMTRF, atol = 1e-5)

    model = ASCNNTRF(16, 128, tmin, tmax, fs, device = device)
    model.loadLTIWeights(trf.weights, trf.bias)
    model = model.eval()
    with torch.no_grad():
        predNNTRF = model(x).cpu().detach().permute(0,2,1).numpy()
    # assert np.allclose(predNNTRF, predNNTRF2, atol = 1e-4)
    # assert np.array_equal(predNNTRF[:, 0:200, :], predNNTRF2[:, 0:200, :])
    # print(predNNTRF[:, 200:300, :], predNNTRF2[:, 200:300, :])
    # assert np.allclose(predNNTRF[:, 200:300, :], predNNTRF2[:, 200:300, :], atol = 1e-5)
    # assert np.array_equal(predNNTRF[:, -100:-1, :], predNNTRF2[:, -100:-1, :])
    # print(predNNTRF[:, -4:, :], predNNTRF2[:, -4:, :])
    # print(predNNTRF, predMTRF)
    # print(predNNTRF.shape, predMTRF.shape)
    # assert np.allclose(predNNTRF[:,:-1,:], predMTRF[:, :-1, :], atol = 1e-6)
    assert np.allclose(predNNTRF2, predNNTRF, atol = 1e-5)
    assert np.allclose(predNNTRF, predMTRF, atol = 1e-5)


def testFuncTRF(basisTRFName, ifFitMTRFWithExtTimeLag = True):
    stimulus, response, fs = load_sample_data(n_segments=9)
    stimulus = [s.mean(axis=1, keepdims=True) for s in stimulus]
    stimulus = stimulus#[:3]
    response = response#[:3]

    #prepare the ASTRF model
    model = ASTRF(1, 128, 0, 700, fs, device = device)
    trfsGen = FuncTRFsGen(1, 128, 0, 700, fs, basisTRFName=basisTRFName, limitOfShift_idx=13, device = device)
    extLagMin, extLagMax = trfsGen.extendedTimeLagRange
    print(extLagMin, extLagMax)
    #prepare the mtrf model
    trf = TRF(direction=1)
    if ifFitMTRFWithExtTimeLag:
        trf.train(stimulus, response, fs, extLagMin/1000, extLagMax/1000, 1000)
    else:
        trf.train(stimulus, response, fs, 0/1000, 700/1000, 1000)

    #train the trfsGen for ASTRF
    trfsGen.fitFuncTRF(trf.weights)
    fig = trfsGen.basisTRF.vis()
    fig.savefig(f'funcTRF_{basisTRFName}.png')
    model.set_trfs_gen(trfsGen)
    model.if_enable_trfsGen = True
    model = model.eval()

    #get mtrf prediction
    nExtTimeLag = trfsGen.limitOfShift_idx
    if ifFitMTRFWithExtTimeLag:
        trf.times = trf.times[nExtTimeLag:-nExtTimeLag]
        trf.weights = trf.weights[:,nExtTimeLag:-nExtTimeLag,:]
    print(trf.weights.shape)
    predMTRF = trf.predict(stimulus)
    predMTRF = np.stack(predMTRF, axis = 0)

    #get ASTRF prediction
    x = torch.stack([torch.tensor(i.T) for i in stimulus], dim = 0).to(device).float()
    nBatch = x.shape[0]
    nSeq = x.shape[2]
    timeinfo = [None for i in range(nBatch)]
    with torch.no_grad():
        predNNTRF = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
    # print(predNNTRF, predMTRF)
    # assert np.allclose(predNNTRF, predMTRF, atol = 1e-1)

    from scipy.stats import pearsonr
    nBatch, nSeq, nChan = predNNTRF.shape
    rs = []
    for b in range(nBatch):
        for c in range(nChan):
            r = pearsonr(predNNTRF[b,:,c], predMTRF[b, :, c])[0]
            rs.append(r)
    print(np.mean(rs))
    assert np.mean(rs) > 0.99


    '''
    test the savemem expr feature
    with torch.no_grad():
        # model.trfsGen.basisTRF.saveMem = True
        predNNTRF2 = model(x, timeinfo).cpu().detach().permute(0,2,1).numpy()
    # print(predNNTRF, predNNTRF2)
    assert np.allclose(predNNTRF, predNNTRF2, atol = 1e-6)
    '''
    

def testTRFEmbed():
    wordsDict = {'he':1, 'is':2, 'a':3, 'old':4, 'man':5, 'who':6, 'has':7, 'been':8, 'fishing':9}

    timeinfo = [
        torch.tensor(
            [
                [1,3,5,7,20,40,45,60],
                [1,3,5,7,20,40,45,60]
            ]
        ).float().to(device),
        torch.tensor(
            [
                [1,3,5,7,45,60],
                [1,3,5,7,45,60]
            ]
        ).float().to(device)
    ]

    model = ASTRF(16, 128, 0, 700, 64, device = device)
    trfsGen = WordTRFEmbedGen(128, 4, 0, 700, 64, wordsDict, device = device)
    model.setTRFsGen(trfsGen)
    model.ifEnableUserTRFGen = True
    model = model.eval()
    words = [
        [
            'he', 'is', 'a', 'old', 'man', 'he', 'is', 'a'
        ],
        [
            'old', 'man', 'who', 'has', 'been', 'fishing'
        ]
    ]
    x = WordTRFEmbedGenTokenizer(wordsDict,  device = device)(words)
    pred = model(x, timeinfo)
    print(pred.shape)
    model.ifEnableUserTRFGen = False
    x = [x_[None, :] for x_ in x]
    pred = model(x, timeinfo)
    print(pred.shape)


# testASCNNTRFLTI(0, 700)
# testASCNNTRFLTI(-100, 300)
# testASCNNTRFLTI(-100, 0)
# testASCNNTRFLTI(0, 300)
# testASTRF()
# testASTRFLTI()
testFuncTRF('gauss', True)
# testFuncTRF('fourier')
# testTRFEmbed()