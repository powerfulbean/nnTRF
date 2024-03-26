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


# will extend to directly remove the existence of oneOfBatch func
# in the future

def seqLast_pad_zero(seq, value = 0):
    maxLen = max([i.shape[-1] for i in seq])
    output = []
    for i in seq:
        output.append(pad(i,(0,maxLen - i.shape[-1]), value = value))
    return torch.stack(output,0)

class CausalConv(torch.nn.Module):

    def __init__(self,inDim,outDim,nKernel,dilation = 1):
        super().__init__()
        self.nKernel = nKernel
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(
            inDim, 
            outDim, 
            nKernel, 
            dilation = dilation
        )
    
    def forward(self,x):
        '''
        x: (nBatch, nChan, nSeq)
        '''
        # padding left
        x = torch.nn.functional.pad(x,(( self.dilation * (self.nKernel-1) ,0)))

        #(nBatch, nOutChan, nSeq)
        x = self.conv(x)
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
        TRFs : TYPE, (nBatch, outDim, nWin, nSeq)
            tensors output by DyTimeEncoder.
        sourceIdx : TYPE, (nBatch, nSeq)
            index of dyImpulse tensor to be assigned to target tensor
        nRealLen: 
            the length of the target
        Returns
        -------
        None.

        '''
        nBatch, outDim, nWin, nSeq = TRFs.shape
        # (nBatch, outDim, nWin, nSeq)
        respUnfold = TRFs 
        maxSrcIdx = torch.max(sourceIdx[:, -1])
        if maxSrcIdx >= nRealLen:
            nRealLen = maxSrcIdx + 1
        # print(outDim,nWin,nRealLen,respUnfold.shape,sourceIdx)
        self.cache = torch.zeros((nBatch, outDim,nWin,nRealLen),device = self.device)

        idxWin = torch.arange(nWin)
        idxChan = torch.arange(outDim)
        idxBatch = torch.arange(nBatch)
        idxWin = idxWin[:, None]
        idxChan = idxChan[:,None, None]
        idxBatch = idxBatch[:, None, None, None]
        sourceIdx = sourceIdx[:,None, None,:]

        self.cache[idxBatch, idxChan, idxWin, sourceIdx] = respUnfold #(nBatch, outDim,nWin,nRealLen)
        self.cache = self.cache.view(nBatch,-1,nRealLen) # (nBatch, outDim*nWin, nRealLen)
        foldOutputSize = (nRealLen + nWin - 1, 1)
        foldKernelSize = (nWin, 1)
        #(nBatch,outDim,foldOutputSize,1)
        output = fold(self.cache,foldOutputSize,foldKernelSize)
        #(nBatch,outDim,nRealLen)
        targetTensor = output[:,:,:nRealLen,0]
        return targetTensor

class LTITRFGen(torch.nn.Module):
    def __init__(self,inDim,nWin,outDim,ifAddBiasInForward = True):
        super().__init__()
        self.inDim = inDim
        self.nWin = nWin
        self.outDim = outDim
        self.weight = torch.nn.Parameter(torch.ones(outDim,inDim,nWin))
        self.bias = torch.nn.Parameter(torch.ones(outDim))
        k = 1 / (inDim * nWin)
        lower = - np.sqrt(k)
        upper = np.sqrt(k)
        torch.nn.init.uniform_(self.weight, a = lower, b = upper)
        torch.nn.init.uniform_(self.bias, a = lower, b = upper)
        self.ifAddBiasInForward = ifAddBiasInForward

    def forward(self,x):
        # x: (nBatch, inDim,nSeq)
        assert x.ndim == 3
        kernelsTemp =  self.weight[None, ..., None] #(1, outDim, inDim, nWin, 1) 
        xTemp = x[:, None, :, None, :] #(nBatch, 1, inDim, 1, nSeq)
        TRFs = xTemp * kernelsTemp  #(nBatch, outDim, inDim, nWin, nSeq)
        if self.ifAddBiasInForward:
            TRFs = TRFs + self.bias[..., None, None, None] #(nBatch, outDim, inDim, nWin, nSeq)
        TRFs = TRFs.sum(2) #(nBatch, outDim, nWin, nSeq)
        return TRFs


class WordTRFEmbedGenTokenizer():
    def __init__(self, wordsDict, device):
        self.wordsDict = wordsDict
        self.device = device

    def __call__(self, words):
        batchTokens = []
        for ws in words:
            tokens = []
            for w in ws:
                tokens.append(self.wordsDict[w])
            batchTokens.append(
                torch.tensor(tokens, dtype = torch.long,device = self.device)
            )
        return batchTokens

class WordTRFEmbedGen(torch.nn.Module):

    def __init__(
        self, 
        outDim,
        hiddenDim,
        tmin_ms,
        tmax_ms,
        fs,
        wordsDict,
        device
    ):
        super().__init__()
        self.outDim = outDim
        self.hiddenDim = hiddenDim
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.fs = fs
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        nWin = len(self.lagTimes)
        self.nWin = nWin
        self.embedding_dim = nWin * hiddenDim

        self.device = device
        self.wordsDict = wordsDict
        self.embedding = torch.nn.Embedding(
            len(wordsDict)+1,
            self.embedding_dim,
            padding_idx = 0
        ).to(device)
        self.proj = torch.nn.Linear(self.hiddenDim, self.outDim, device = device)

    def forward(self, batchTokens):
        # (nBatch, outDim, nWin, nSeq)
        batchTokens = seqLast_pad_zero(batchTokens)
        # (nBatch, nWin * hiddenDim)
        trfs = self.embedding(batchTokens)
        # print(trfs.shape)
        trfs = trfs.reshape(*trfs.shape[:2], self.hiddenDim, self.nWin)
        # (nBatch, nSeq, nWin, hiddenDim)
        # print(trfs.shape)
        trfs = trfs.permute(0, 1, 3, 2)
        # (nBatch, nSeq, nWin, outDim)
        # print(torch.cuda.memory_allocated()/1024/1024)
        trfs = self.proj(trfs)
        # (nBatch, outDim, nWin, nSeq)
        trfs = trfs.permute(0, 3, 2, 1)
        # print(trfs.shape)
        return trfs 


class FourierBasisTRF(torch.nn.Module):
    
    def __init__(self,nInChan,nOutChan,nLag,nBasis,device = 'cpu'):
        #TRFs the TRF for some channels
        super().__init__()
        self.nBasis = nBasis
        self.nInChan = nInChan
        self.nOutChan = nOutChan
        self.nLag = nLag
        coefs = torch.empty((nOutChan, nInChan, nBasis),device=device)
        TRFs = torch.empty((nOutChan, nInChan, nLag),device=device)
        self.register_buffer('coefs', coefs)
        self.register_buffer('TRFs',TRFs)
        self.T = self.nLag - 1
        self.device = device
        maxN = nBasis // 2
        # (maxN)
        self.seqN = torch.arange(1,maxN+1,device = self.device)
        self.saveMem = False

    def fitTRFs(self,TRFs):
        TRFs = torch.from_numpy(TRFs)
        TRFs = TRFs.permute(2, 0, 1)
        self.TRFs[:,:,:] = TRFs.to(self.device)[:,:,:]
        fd_basis_s = []
        grid_points = list(range(self.nLag))
        for j in range(self.nOutChan):
            for i in range(self.nInChan):
                TRF = TRFs[j, i, :]
                fd = skfda.FDataGrid(
                    data_matrix=TRF,
                    grid_points=grid_points,
                )
                basis = skfda.representation.basis.Fourier(n_basis = self.nBasis)
                fd_basis = fd.to_basis(basis)
                coef = fd_basis.coefficients[0]
                self.coefs[j, i, :] = torch.from_numpy(coef).to(self.device)
                
                T = fd_basis.basis.period
                assert T == self.T
                fd_basis_s.append(fd_basis)
                
        out = self.vecFourierSum(
            self.nBasis,
            self.T,
            torch.arange(0,self.nLag).view(1,1,1,-1,1).to(self.device),
            self.coefs
        )
        out = out[0][...,0]
        for j in range(self.nOutChan):
            for i in range(self.nInChan):
                fd_basis = fd_basis_s[j*self.nInChan + i]
                temp = fd_basis(np.arange(0,self.nLag)).squeeze()
                curFTRF = out[j, i,:].cpu().numpy()
                TRF = TRFs[j, i,:]
                assert np.around(pearsonr(TRF, temp)[0]) >= 0.99
                try:
                    assert np.allclose(curFTRF,temp,atol = 1e-6)
                except:
                    print(TRF, curFTRF,temp)
                    raise 
                # print(i,j,pearsonr(TRF, curFTRF))
                
    
    def phi0(self,T):
        return 1 / ((2 ** 0.5) * ((T/2) ** 0.5))

    def phi2n_1(self,n,T,t):
        #n: (maxN)
        #t: (nBatch, 1, 1, nWin, nSeq, 1)

        #(nBatch, 1, 1, nWin, nSeq, maxN)
        t_input = 2 * torch.pi * t * n / T
        #(nBatch, 1, 1, nSeq, maxN, nWin)
        t_input = t_input.permute(0, 1, 2, 4, 5, 3)
        signal = torch.sin(t_input) / (T/2)**0.5
        return signal.permute(0, 1, 2, 5, 3, 4)
    
    def phi2n(self,n,T,t):
        #n: (maxN)
        #t: (nBatch, 1, 1, nWin, nSeq, 1)
        #(nBatch, 1, 1, nWin, nSeq, maxN)
        t_input = 2 * torch.pi * t * n / T
        #(nBatch, 1, 1, nSeq, maxN, nWin)
        t_input = t_input.permute(0, 1, 2, 4, 5, 3)
        signal = torch.cos(t_input) / (T/2)**0.5
        return signal.permute(0, 1, 2, 5, 3, 4)
    
    def vecFourierSum(self,nBasis, T, t,coefs):
        #coefs: (nOutChan, nInChan, nBasis)
        #t: (nBatch, 1, 1, nWin, nSeq)
        #   if tChan of t is just 1, which means we share
        #   the same time-axis transformation for all channels
        #return: (nBatch, outDim, inDim, nWin, nSeq)

        #(nBatch, 1, 1, nWin, nSeq, 1)
        t = t[..., None]
        const0 = self.phi0(T)
        maxN = nBasis // 2
        # (maxN)
        seqN = self.seqN
        # (nBatch, 1, 1, nWin, nSeq, maxN)
        constSin = self.phi2n_1(seqN, T, t) 
        constCos = self.phi2n(seqN, T, t)

        # (nBatch, 1, 1, nWin, nSeq, 2 * maxN)
        constN = torch.stack(
            [constSin,constCos],
            axis = -1
        ).reshape(*constSin.shape[:5], 2*maxN)
        # print(const0,[i.shape for i in [constN, coefs]])
        memAvai,_ = torch.cuda.mem_get_info()

        nBatch, _, _, nWin, nSeq, nBasis =  constN.shape
        nOutChan, nInChan, nBasis = coefs.shape

        #(nOutChan, nInChan, 1, 1, nBasis)
        coefs = coefs[:, :, None, None, :]
        nBasis = nBasis
        nMemReq = nBatch * nSeq * nInChan * nOutChan * nBasis * nWin * 4 # 4 indicates 4 bytes
        # print(torch.cuda.memory_allocated()/1024/1024)
        if nMemReq > memAvai * 0.9 or self.saveMem:
            out = const0 * coefs[...,0] #(nOutChan, nInChan, 1, 1)
            for nB in range(2 * maxN):
                out = out + constN[...,nB] * coefs[...,1+nB]
        else:
            # (nbatch, nOutChan, nInChan, nWin, nSeq, nBasis)
            out =  const0 * coefs[...,0] + (constN * coefs[...,1:]).sum(-1)
        
        # print(torch.cuda.memory_allocated()/1024/1024)
        # (nBatch, outDim, inDim, nWin, nSeq)
        return out
    
    def forward(self,x):
        # x: (nBatch, 1, 1, nWin, nSeq)\
        # return: 
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
            torch.arange(0,self.nLag).view(1,1,1,-1,1).to(self.device),
            self.coefs
        )[0][...,0]
        for j in range(nOutChan):
            for i in range(nInChan):
                TRF = self.TRFs[j,i,:].cpu()
                FTRF = FTRFs[j,i,:].cpu()
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
        transformer = None,
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
        self.transformer = transformer
        self.nMiddleParam = len(mode.split(','))
        self.device = device
        self.basisTRF = FourierBasisTRF(
            inDim, 
            outDim, 
            nWin + 2 * limitOfShift_idx, 
            nBasis,
            device=device
        )

        if transformer is None:
            inDim, outDim, device = self.expectedInOutDimOfFeatExtracter()
            self.transformer = CausalConv(inDim, outDim, 2).to(device)

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
            self.basisTRF.fitTRFs(w)
        return self
    
    def pickParam(self,paramSeqs,idx):
        #paramSeqs: (nBatch, nMiddleParam, 1, 1, nSeq)
        return paramSeqs[:, idx:idx+1, ...]
    
    def getTransformParams(self, x, startIdx = None):
        paramSeqs = self.transformer(x) #(nBatch, nMiddleParam, nSeq)
        if startIdx is not None:
            nBatch, nMiddleParam, _ = paramSeqs.shape
            idxBatch = torch.arange(nBatch)
            idxMiddleParam = torch.arange(nMiddleParam)
            idxMiddleParam = idxMiddleParam[:, None]
            startIdx = startIdx[:, None, :]
            idxBatch = idxBatch[:, None, None]
            paramSeqs = paramSeqs[idxBatch, idxMiddleParam, startIdx]
        return paramSeqs

    def forward(self, x, startIdx = None):
        '''
        x: (nBatch, inDim, nSeq)
        output: TRFs (nBatch, outDim, nWin, nSeq)
        '''
        #(nBatch, nMiddleParam, nSeq)
        paramSeqs = self.getTransformParams(x, startIdx)
        # print(paramSeqs.shape)

        nBatch, nMiddleParam, nSeq= paramSeqs.shape

        #(nBatch, nMiddleParam, 1, 1, nSeq)
        paramSeqs = paramSeqs[:, :, None, None, :]
        midParamList = self.mode.split(',')
        if midParamList == ['']:
            midParamList = []
        nParamMiss = 0
        if 'a' in midParamList:
            aIdx = midParamList.index('a')
            #(nBatch, 1, 1, 1, nSeq)
            aSeq = self.pickParam(paramSeqs, aIdx) 
            aSeq = torch.abs(aSeq)
        elif '+-a' in midParamList:
            aIdx = midParamList.index('+-a')
            #(nBatch, 1, 1, 1, nSeq)
            aSeq = self.pickParam(paramSeqs, aIdx)
        else:
            nParamMiss += 1
            #(nBatch, 1, inDim, 1, nSeq)
            aSeq = x[:, None, :, None, :]
        
        if 'b' in midParamList:
            bIdx = midParamList.index('b')
            #(nBatch, 1, 1, 1, nSeq)
            bSeq = self.pickParam(paramSeqs, bIdx) 
            bSeq = torch.maximum(bSeq, - self.limitOfShift_idx)
            bSeq = torch.minimum(bSeq,   self.limitOfShift_idx)
        else:
            nParamMiss += 1
            bSeq = 0
            
        if 'c' in midParamList:
            cIdx = midParamList.index('c')
            #(nBatch, 1, 1, 1, nSeq)
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
        #(1, 1, 1, nWin, 1) 
        nSeq = self.lagIdxs_ts[None, None, None, :, None] + self.limitOfShift_idx 
        
        #(nBatch, outDim, inDim, nWin, nSeq)
        nonLinTRFs = aSeq * self.basisTRF( cSeq * ( nSeq -  bSeq) ) 
        # print(torch.cuda.memory_allocated()/1024/1024)

        #(nBatch, outDim, nWin, nSeq)
        TRFs = nonLinTRFs.sum(2)
        # print(torch.cuda.memory_allocated()/1024/1024)
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
        device = 'cpu',
        mode_time_impulse = False
    ):
        super().__init__()
        assert tmin_ms >= 0
        self.mode_time_impulse = mode_time_impulse
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
        w = w.permute(2, 0, 1) #(nOutChan, nInChan, nLag)
        with torch.no_grad():
            self.ltiTRFsGen.weight = torch.nn.Parameter(w)
            self.ltiTRFsGen.bias = torch.nn.Parameter(b)
        return self

    def exportLTIWeights(self):
        with torch.no_grad():
            # (nInChan, nLag, nOutChan)
            w = self.ltiTRFsGen.weight.cpu().detach().permute(1, 2, 0).numpy()
            b = self.ltiTRFsGen.bias.cpu().detach().numpy().reshape(1,-1)
        w = w * self.fs 
        b = b * self.fs
        return w, b

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
        #X: nBatch * [nChan, nSeq]
        #timeinfo: nBatch * [2, nSeq]

        nSeqs = []
        nRealLens = []
        trfStartIdxs = []
        for ix, xi in enumerate(x):
            nLenXi = xi.shape[-1]
            if timeinfo[ix] is not None:
                # print(timeinfo[ix].shape)
                if not self.mode_time_impulse:
                    assert timeinfo[ix].shape[-1] == xi.shape[-1]
                nLen = torch.ceil(timeinfo[ix][0][-1] * self.fs).long() + self.nWin
                startIdx = torch.round(timeinfo[ix][0,:] * self.fs).long() + self.lagIdxs[0]
            else:
                nLen = nLenXi
                startIdx = torch.tensor(np.arange(nLen)) + self.lagIdxs[0]
            nSeqs.append(nLenXi)
            nRealLens.append(nLen)
            trfStartIdxs.append(startIdx)

        nGlobLen = max(nRealLens)
        x = seqLast_pad_zero(x)
        trfStartIdxs = seqLast_pad_zero(trfStartIdxs, value = -1)
        #(nBatch, outDim, nWin, nSeq)
        if self.mode_time_impulse:
            TRFs = self.getTRFs(x, trfStartIdxs)
        else:
            TRFs = self.getTRFs(x)

        ##(nBatch,outDim,nRealLen)
        targetTensor = self.trfAligner(TRFs,trfStartIdxs,nGlobLen)

        if self.ifEnableUserTRFGen:
            targetTensor = targetTensor + self.bias.view(-1,1)
        else:
            ltiTRFBias = self.ltiTRFsGen.bias
            targetTensor = targetTensor + ltiTRFBias.view(-1,1)

        return targetTensor
    
    def getTRFs(self, x, startIdx = None):
        if self.ifEnableUserTRFGen:
            return self.trfsGen(x, startIdx)
        else:
            return self.ltiTRFsGen(x)

class ASCNNTRF(ASTRF):
    #perform CNNTRF within intervals,
    #and change the weights

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
        torch.nn.Module.__init__(self)
        # assert tmin_ms >= 0
        self.inDim = inDim
        self.outDim = outDim
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        self.tmin_idx = self.lagIdxs[0]
        self.tmax_idx = self.lagIdxs[-1]
        nWin = len(self.lagTimes)
        self._nWin = nWin
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
        
        self._enableUserTRFGen = False 
        self.device = device

    @property
    def nWin(self):
        return self._nWin
    
    def getTRFs(self, ctx):
        # ctx: (nBatch, inDim,nSeq)
        # #nTRFs is the number of TRFs needed

        #note the difference between the ASTRF and ASCNNTRF at here
        #the LTITRF is not multiplied with x

        
        if self.ifEnableUserTRFGen:
            #how to decide how much trfs to return????
            TRFs = self.trfsGen(ctx)
            #TRFs (nBatch, outDim, nWin, nSeq)
            TRFs = TRFs[0]  #(outDim, nWin, nSeq)
            TRFs = TRFs.permute(2, 0, 1)[..., None, :]
        else:
            #ctx: (nBatch, inDim,nSeq) (nTRFs, inDim,nSeq)
            nTRFs = ctx.shape[-1]#len(ctx)
            # TRFs nTRFs * (nChanOut, nChanIn, nWin)
            TRFs = [self.ltiTRFsGen.weight] * nTRFs
        return TRFs

    def getTRFSwitchOnsets(self, x):
        #X: nBatch * [nChan, nSeq]
        return [0, 200, 300, 400, 500, 600, 800, 1000]


    def defaultCtx(self, switchOnsets, x):
        switchOnsets2 = switchOnsets + [-1]
        ctx = []
        for i in range(len(switchOnsets)):
            ctx.append(x[:, :, switchOnsets2[i]:switchOnsets2[i+1]])
        return ctx

    def forward(self, x, timeInfo = None, ctx = None):
        
        #currently only support single batch

        #X: nBatch * [nChan, nSeq]
        #timeinfo: nBatch * [2, nSeq]
        if timeInfo is None:
            TRFSwitchOnsets = self.getTRFSwitchOnsets(x)
        else:
            TRFSwitchOnsets = timeInfo
        if ctx is None:
            ctx = self.defaultCtx(TRFSwitchOnsets, x)
        nTRFs = len(TRFSwitchOnsets)
        nBatch, nChan, nSeq = x.shape
        TRFs = self.getTRFs(ctx)
        TRFsFlip = [TRF.flip([-1]) for TRF in TRFs]
        # print([torch.equal(TRFsFlip[0], temp) for temp in TRFsFlip[1:]])
        # print(TRFsFlip)
        TRFSwitchOnsets.append(None)

        nPaddedOutput = nSeq \
            + max(-self.tmin_idx, 0) \
            + max( self.tmax_idx, 0)

        output = torch.zeros(
            nBatch, 
            self.outDim, 
            nPaddedOutput, 
            device = self.device
        )

        #segment startIdx offset
        startOffset = self.tmin_idx
        #global startIdx offset
        offset = min(0, self.tmin_idx)

        realOffset = startOffset - offset

        for idx, TRFFlip in enumerate(TRFsFlip):
            t_start = TRFSwitchOnsets[idx]
            t_end = TRFSwitchOnsets[idx+1]
            t_x = x[..., t_start : t_end]
            segment = self.trfCNN(t_x,TRFFlip)
            # print(realOffset)
            t_startReal = t_start + realOffset
            t_endReal = t_startReal + segment.shape[-1]
            # print(segment.shape, t_startReal, t_endReal, t_x.shape, x.shape)
            output[:,:,t_startReal : t_endReal] += segment 

       #decide how to crop the output based on tmin and tmax
        startIdx = max(-self.tmin_idx, 0)
        lenOutput = output.shape[-1]
        endIdx   = lenOutput - max( self.tmax_idx, 0)
        
        return output[..., startIdx: endIdx] + self.ltiTRFsGen.bias.view(-1, 1)

    def trfCNN(self, x, TRFFlip):
        #X: (nBatch, nChan, nSubSeq)
        # TRFsFlip is the TRFs with its kernel dimension flipped 
        # for Conv1D!
        
        #need to first padding
        #timelag doesn't influence how much to pad
        #  but the offset of the startIdx
        nWin = TRFFlip.shape[-1]
        x = torch.nn.functional.pad(x, (nWin-1, nWin-1))
        #then do the conv
        output = torch.nn.functional.conv1d(x, TRFFlip)
        return output

