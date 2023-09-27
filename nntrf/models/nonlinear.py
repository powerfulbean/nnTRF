import math
import numpy as np
import torch
import skfda
from scipy.stats import pearsonr
from torch.nn.functional import pad, fold
from .linear import msec2Idxs, Idxs2msec
try:
    from matplotlib import pyplot as plt
except:
    plt = None

def seqLast_pad_zero(seq):
    maxLen = max([i.shape[-1] for i in seq])
    output = []
    for i in seq:
        output.append(pad(i,(0,maxLen - i.shape[-1])))
    return torch.stack(output,0)

class CausalConv(torch.nn.Module):

    def __init__(self,nInChan,nOutChan,nKernel,dilation = 1):
        super().__init__()
        self.nKernel = nKernel
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(
            nInChan, 
            nOutChan, 
            nKernel, 
            dilation = dilation
        )
    
    def forward(self,x):
        '''
        x: (nChan, nSeq)
        '''
        # padding buttom
        x = torch.nn.functional.pad(x,(( self.dilation * (self.nKernel-1) ,0)))
        x = self.conv(x)
        x = x.swapaxes(-1,-2) # (nSeq, nOut)
        return x

class TRFAligner(torch.nn.Module):
    
    def __init__(self,device):
        super().__init__()
        self.device = device

    def forward(self,TRFs,sourceIdx,nRealLen):#,targetTensor):
        '''
        in-place operation
        Parameters
        ----------
        TRFs : TYPE, (nSeq, nWin, outDim)
            tensors output by DyTimeEncoder.
        sourceIdx : TYPE, (nSeq * srate)
            index of dyImpulse tensor to be assigned to target tensor
        nRealLen: 
            the length of the target
        Returns
        -------
        None.

        '''
        nWin = TRFs.shape[1]
        outDim = TRFs.shape[2]
        # (outDim,  nWin, nSeq)
        respUnfold = TRFs.permute((2,1,0)) #TRFs.T #torch.unsqueeze(scrchTRFs,-1) 
        if sourceIdx[-1] >= nRealLen:
            nRealLen = sourceIdx[-1] + 1
        # print(outDim,nWin,nRealLen,respUnfold.shape,sourceIdx)
        self.cache = torch.zeros((outDim,nWin,nRealLen),device = self.device)
        self.cache[:,:,sourceIdx] = respUnfold #(outDim,nWin,nRealLen)
        self.cache = self.cache.unsqueeze(0) #(1,outDim, nWin, nRealLen)
        self.cache = self.cache.view(1,-1,nRealLen) # (1, outDim*nWin, nRealLen)
        foldOutputSize = (nRealLen + nWin - 1, 1)
        foldKernelSize = (nWin, 1)
        #(1,outDim,foldOutputSize,1)
        output = fold(self.cache,foldOutputSize,foldKernelSize)
        targetTensor = output[0,:,:nRealLen,0]
        return targetTensor

class LTITRFGen(torch.nn.Module):
    def __init__(self,inDim,nWin,outDim,ifAddBiasInForward = True):
        super().__init__()
        self.inDim = inDim
        self.nWin = nWin
        self.outDim = outDim
        self.weight = torch.nn.Parameter(torch.ones(inDim,nWin,outDim))
        self.bias = torch.nn.Parameter(torch.ones(outDim))
        k = 1 / (inDim * nWin)
        lower = - np.sqrt(k)
        upper = np.sqrt(k)
        torch.nn.init.uniform_(self.weight, a = lower, b = upper)
        torch.nn.init.uniform_(self.bias, a = lower, b = upper)
        self.ifAddBiasInForward = ifAddBiasInForward

    def forward(self,x):
        # x: (inDim,nSeq)
        kernelsTemp =  self.weight.unsqueeze(0) #(1, inDim, nWin,outDim)
        xTemp = x.T.view(-1,self.inDim,1,1) #(nSeq, inDim, 1, 1) 
        TRFs = xTemp * kernelsTemp 
        if self.ifAddBiasInForward:
            TRFs = TRFs + self.bias #(nSeq,inDim,nWin,outDim)
        TRFs = TRFs.sum(1)
        return TRFs

class FourierFuncTRF(torch.nn.Module):
    
    def __init__(self,nInChan,nOutChan,nLag,nBasis,device = 'cpu'):
        #TRFs the TRF for some channels
        super().__init__()
        self.nBasis = nBasis
        self.nInChan = nInChan
        self.nOutChan = nOutChan
        self.nLag = nLag
        coefs = torch.empty((nInChan,nOutChan,nBasis),device=device)
        TRFs = torch.empty((nInChan,nLag,nOutChan),device=device)
        self.register_buffer('coefs', coefs)
        self.register_buffer('TRFs',TRFs)
        self.T = self.nLag - 1
        self.device = device
        maxN = nBasis // 2
        # (maxN,1)
        self.seqN = torch.arange(1,maxN+1,device = self.device).reshape(-1,1)
        self.saveMem = False

    def fitTRFs(self,TRFs):
        self.TRFs[:,:,:] = torch.from_numpy(TRFs).to(self.device)[:,:,:]
        fd_basis_s = []
        grid_points = list(range(self.nLag))
        for i in range(self.nInChan):
            for j in range(self.nOutChan):
                TRF = TRFs[i,:,j]
                fd = skfda.FDataGrid(
                    data_matrix=TRF,
                    grid_points=grid_points,
                )
                basis = skfda.representation.basis.Fourier(n_basis = self.nBasis)
                fd_basis = fd.to_basis(basis)
                coef = fd_basis.coefficients[0]
                self.coefs[i,j,:] = torch.from_numpy(coef).to(self.device)
                
                T = fd_basis.basis.period
                assert T == self.T
                fd_basis_s.append(fd_basis)
                
        out = self.vecFourierSum(
            self.nBasis,
            self.T,
            torch.arange(0,self.nLag).view(1,1,1,-1).to(self.device),
            self.coefs
        )[0]
        for i in range(self.nInChan):
            for j in range(self.nOutChan):
                fd_basis = fd_basis_s[i*self.nOutChan + j]
                temp = fd_basis(np.arange(0,self.nLag)).squeeze()
                curFTRF = out[i,:,j].cpu().numpy()
                TRF = TRFs[i,:,j]
                assert np.allclose(curFTRF,temp,atol = 1e-6)
                # print(i,j,pearsonr(TRF, curFTRF))
                assert np.around(pearsonr(TRF, curFTRF)[0]) >= 0.99
    
    def phi0(self,T):
        return 1 / ((2 ** 0.5) * ((T/2) ** 0.5))

    def phi2n_1(self,n,T,t):
        return torch.sin(2 * torch.pi * t * n / T) / (T/2)**0.5
    
    def phi2n(self,n,T,t):
        return torch.cos(2 * torch.pi * t * n / T) / (T/2)**0.5
    
    def vecFourierSum(self,nBasis, T, t,coefs):
        #coefs: (nInChan,nOutChan,nBasis)
        #t: (nSeq, tInChan, tOutChan, nLag) 
        #if tChan of t is just 1, which means we share
        #the same time-axis transformation for all channels

        t = t.unsqueeze(-2) #(nSeq, tInChan, tOutChan,1,nLag)
        coefs = coefs.unsqueeze(-1) #(nInChan,nOutChan,nBasis,1)
        const0 = self.phi0(T)
        maxN = nBasis // 2
        # seqN = torch.arange(1,maxN+1,device = self.device).reshape(-1,1) 
        # (maxN,1)
        seqN = self.seqN
        constSin = self.phi2n_1(seqN,T,t) # (nSeq, nInChan, nOutChan, maxN, nLag)
        constCos = self.phi2n(seqN, T, t) # (nSeq, nInChan, nOutChan, maxN, nLag)

        # (nSeq, nInChan, nOutChan, maxN * 2, nLag)
        constN = torch.stack(
            [constSin,constCos],
            axis = -2
        ).reshape(*t.shape[0:3],2*maxN,-1)
        # print(const0,[i.shape for i in [constN, coefs]])
        memAvai,_ = torch.cuda.mem_get_info()
        nSeq, nInChan, _, nBasis, nLag =  constN.shape
        nBasis = nBasis + 1
        nInChan,nOutChan,nBasis,_ = coefs.shape
        nMemReq = nSeq * nInChan * nOutChan * nBasis * nLag * 4
        # print(torch.cuda.memory_allocated()/1024/1024)
        if nMemReq > memAvai * 0.9 or self.saveMem:
            out = const0 * coefs[...,0,:]
            for nB in range(2 * maxN):
                out = out + constN[...,nB,:] * coefs[...,1+nB,:]
        else:
            # constN = constN.unsqueeze(1).unsqueeze(1)
            # (nSeq,nInChan,nOutChan, nLag)
            out =  const0 * coefs[...,0,:] + (constN * coefs[...,1:,:]).sum(-2)
        
        # print(torch.cuda.memory_allocated()/1024/1024)
        return out.permute(0,1,3,2)
    
    def forward(self,x):
        # output = torch.empty(self.nInChan,self.nLag,self.nOutChan)
        coefs = self.coefs
        out = self.vecFourierSum(self.nBasis,self.T,x,coefs)
        return out
        
    def visResult(self):
        if plt is None:
            raise ValueError('matplotlib should be installed')
        fig, axs = plt.subplots(2)
        fig.suptitle('top: original TRF, bottom: reconstructed TRF')
        nInChan = self.nInChan
        nOutChan = self.nOutChan
        FTRFs = self.vecFourierSum(
            self.nBasis,
            self.T,
            torch.arange(0,self.nLag).view(1,1,1,-1).to(self.device),
            self.coefs
        )[0]
        for i in range(nInChan):
            for j in range(nOutChan):
                TRF = self.TRFs[i,:,j].cpu()
                FTRF = FTRFs[i,:,j].cpu()
                axs[0].plot(TRF)
                axs[1].plot(FTRF)
        return fig

class FuncTRFsGen(torch.nn.Module):
    '''
    Implement the functional TRF generator, generate dynamically 
        warped TRF by transform the functional TRF template
    '''

    def __init__(
        self, 
        inDim, 
        outDim,
        tmin_ms,
        tmax_ms,
        fs,
        limitOfShift_idx = 7,
        nBasis = 21, 
        mode = '',
        featExtracter = None,
        auxInDim = 0,
        device = 'cpu'
    ):
        super().__init__()
        assert mode.replace('+-','') in ['','a','b','a,b','a,b,c']
        self.inDim = inDim
        self.auxInDim = auxInDim
        self.outDIm = outDim
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.fs = fs
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagIdxs_ts = torch.Tensor(self.lagIdxs).float().to(device)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        nWin = len(self.lagTimes)
        self.nWin = nWin
        self.limitOfShift_idx = limitOfShift_idx
        self.nBasis = nBasis
        self.mode = mode
        self.featExtracter = featExtracter
        self.nMiddleParam = len(mode.split(','))
        self.device = device
        self.funcTRF = FourierFuncTRF(
            inDim, 
            outDim, 
            nWin + 2 * limitOfShift_idx, 
            nBasis,
            device=device
        )

        if featExtracter is None:
            inDim, outDim, device = self.expectedInOutDimOfFeatExtracter()
            self.featExtracter = CausalConv(inDim, outDim, 2).to(device)

        self.limitOfShift_idx = torch.tensor(limitOfShift_idx)

    @property
    def extendedTimeLagRange(self):
        minLagIdx = self.lagIdxs[0]
        maxLagIdx = self.lagIdxs[-1]
        left = np.arange(minLagIdx - self.limitOfShift_idx, minLagIdx)
        right = np.arange(maxLagIdx + 1, maxLagIdx + 1 + self.limitOfShift_idx)
        extLag_idx = np.concatenate([left, self.lagIdxs, right])
        assert len(extLag_idx) == self.nWin + 2 * self.limitOfShift_idx
        timelags = Idxs2msec(extLag_idx, self.fs)
        return timelags[0], timelags[-1]

    def expectedInOutDimOfFeatExtracter(self):
        inDim = self.inDim + self.auxInDim
        device = self.device
        outDim = self.inDim * self.nMiddleParam
        return inDim, outDim, device

    def fitFuncTRF(self, w):
        w = w * 1 / self.fs
        with torch.no_grad():
            self.funcTRF.fitTRFs(w)
        return self
    
    def pickParam(self,paramSeqs,idx):
        #paramSeqs: (nSeq, nChan)
        return paramSeqs[...,idx:idx+1]
    
    def forward(self, x):
        '''
        x: (inDim,nSeq)
        '''
        paramSeqs = self.featExtracter(x)
        nSeq, nChan = paramSeqs.shape
        #(nSeq, self.inDim, 1, nMiddleParam)
        paramSeqs = paramSeqs.view(nSeq,self.inDim,1,-1) 
        # nBatch,nSeq,nChan = paramSeqs.shape
        midParamList = self.mode.split(',')
        if midParamList == ['']:
            midParamList = []
        nParamMiss = 0
        if 'a' in midParamList:
            aIdx = midParamList.index('a')
            aSeq = self.pickParam(paramSeqs, aIdx) #(nSeq,self.inDim,1,1)
            # aSeq = aSeq.permute(0,1,3,2) #(nSeq,self.inDim,1,1)
            aSeq = torch.abs(aSeq)
        elif '+-a' in midParamList:
            aIdx = midParamList.index('+-a')
            aSeq = self.pickParam(paramSeqs, aIdx) #(nSeq,self.inDim,1,1)
            # aSeq = aSeq.permute(0,1,3,2) #(nSeq,self.inDim,1,1)
        else:
            nParamMiss += 1
            aSeq = x.T 
            aSeq = aSeq.view(*aSeq.shape, 1, 1) #: (nSeq,self.inDim,1,1) 
        
        if 'b' in midParamList:
            bIdx = midParamList.index('b')
            bSeq = self.pickParam(paramSeqs, bIdx) #(nSeq,self.inDim,1,1)
            bSeq = torch.maximum(bSeq, - self.limitOfShift_idx)
            bSeq = torch.minimum(bSeq,   self.limitOfShift_idx)
        else:
            nParamMiss += 1
            bSeq = 0
            
        if 'c' in midParamList:
            cIdx = midParamList.index('c')
            cSeq = self.pickParam(paramSeqs, cIdx)
            #two reasons, cSeq must be larger than 0; 
            #if 1 is the optimum, abs will have two x for the optimum, 
            # which is not stable 
            cSeq =  1 + cSeq
            cSeq = torch.maximum(cSeq, torch.tensor(0.5))
            cSeq = torch.minimum(cSeq, torch.tensor(1.4))
        else:
            nParamMiss += 1
            cSeq = 1

        assert (len(midParamList) + nParamMiss) == 3
        nSeq = self.lagIdxs_ts.view(1,1,1,-1) + self.limitOfShift_idx 
        #(1,1,1,nLag)
        
        nonLinTRFs = aSeq * self.funcTRF( cSeq * ( nSeq -  bSeq) ) 
        #(nSeq,self.inDim,nLag,self.outDim)
        print(torch.cuda.memory_allocated()/1024/1024)
        TRFs = nonLinTRFs.sum(1) #(nSeq,nLag,self.outDim)
        print(torch.cuda.memory_allocated()/1024/1024)
        return TRFs

class ASTRF(torch.nn.Module):
    '''
    the TRF implemented the convolution sum of temporal response,
        (i.e., time-aligning the temporal responses at their 
        corresponding location, and point-wise sum them).
        It requres a module to generate temproal responses to each
        individual stimuli, and also require time information to
        displace/align the temporal responses at the right
        indices/location 

    limitation: can't do TRF for zscored input, under this condition
      location with no stimulus will be non-zero.
    '''

    def __init__(
        self,
        inDim,
        outDim,
        tmin_ms,
        tmax_ms,
        fs,
        trfsGen = None,
        device = 'cpu'
    ):
        super().__init__()
        assert tmin_ms >= 0
        self.inDim = inDim
        self.outDim = outDim
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        nWin = len(self.lagTimes)
        self.nWin = nWin
        self.ltiTRFsGen = LTITRFGen(
            inDim,
            nWin,
            outDim,
            ifAddBiasInForward=False
        ).to(device)
        self.trfsGen = trfsGen if trfsGen is None else trfsGen.to(device)
        self.fs = fs

        self.bias = None
        #also train bias for the trfsGen provided by the user
        if self.trfsGen is not None:
            self.bias = torch.nn.Parameter(torch.ones(outDim, device = device))
            fan_in = inDim * nWin
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
        self.trfAligner = TRFAligner(device)
        self._enableUserTRFGen = False 
        self.device = device

    def setTRFsGen(self, trfsGen):
        self.trfsGen = trfsGen.to(self.device)
        self.bias = torch.nn.Parameter(
            torch.ones(self.outDim, device = self.device)
        )
        fan_in = self.inDim * self.nWin
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def getParamsForTrain(self):
        raise NotImplementedError()
        return [i for i in self.oNonLinear.parameters()] + [self.bias]

    def loadLTIWeights(self, w, b):
        #w: (nInChan, nLag, nOutChan)
        b = b[0]
        w = w * 1 / self.fs
        b = b * 1/ self.fs
        w = torch.FloatTensor(w).to(self.device)
        b = torch.FloatTensor(b).to(self.device)
        
        with torch.no_grad():
            self.ltiTRFsGen.weight = torch.nn.Parameter(w)
            self.ltiTRFsGen.bias = torch.nn.Parameter(b)
        return self

    @property
    def ifEnableUserTRFGen(self):
        return self._enableUserTRFGen

    @ifEnableUserTRFGen.setter
    def ifEnableUserTRFGen(self,x):
        assert isinstance(x, bool)
        print('set ifEnableNonLin',x)
        if x == True and self.trfsGen is None:
            raise ValueError('trfGen is None, cannot be enabled')
        self._enableUserTRFGen = x

    def stopUpdateLinear(self):
        self.ltiTRFsGen.weight.requires_grad_(False)
        self.ltiTRFsGen.weight.grad = None
        self.ltiTRFsGen.bias.grad = None
        self.ltiTRFsGen.bias.requires_grad_(False)
        
    def enableUpdateLinear(self):
        self.ltiTRFsGen.requires_grad_(True)
        self.ltiTRFsGen.weight.grad = torch.zeros_like(self.ltiTRFsGen.weight)
        self.ltiTRFsGen.bias.grad = torch.zeros_like(self.ltiTRFsGen.bias)

    def forward(self, x, timeinfo):
        output = list()
        for idx,_ in enumerate(x):
            #need to do this for every batch because 
            # (I am lazy) they have different tIntvl Information
            predForBatch = self.oneOfBatch(x[idx],timeinfo[idx])
            output.append(predForBatch)
        return seqLast_pad_zero(output)#torch.stack(output,0)

    def oneOfBatch(self, x, timeinfo):
        if timeinfo is not None:
            assert timeinfo.ndim == 2
            assert timeinfo.shape[0] ==2
            assert timeinfo.shape[1] == x.shape[1]
            nLen = torch.ceil(timeinfo[0][-1] * self.fs).long() + self.nWin
            vIdxStart = torch.round(timeinfo[0,:] * self.fs).long()
        else:
            nLen = x.shape[1]
            vIdxStart = torch.tensor(np.arange(nLen))
        vIdxEnd = None #torch.round(tIntvl[1,:] * self.fs).long()
        # ltiTRF = self.ltiTRFsGen.weight #(inDim, nWin,outDim)
        ltiTRFBias = self.ltiTRFsGen.bias
        
        TRFs = self.getTRFs(x)
        sourceIdx = vIdxStart + self.lagIdxs[0]
        targetTensor = self.trfAligner(TRFs,sourceIdx,nLen)

        if self.ifEnableUserTRFGen:
            targetTensor = targetTensor + self.bias.view(-1,1)
        else:
            targetTensor = targetTensor + ltiTRFBias.view(-1,1)

        return targetTensor
    
    def getTRFs(self, x):
        if self.ifEnableUserTRFGen:
            return self.trfsGen(x)
        else:
            return self.ltiTRFsGen(x)


