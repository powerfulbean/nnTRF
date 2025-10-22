import math
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.nn.functional import pad, fold
from .linear import msec2Idxs, Idxs2msec, CPadOrCrop1D
try:
    import skfda
except:
    skfda = None

try:
    from matplotlib import pyplot as plt
except:
    plt = None

try:
    from mtrf.model import TRF
except:
    TRF = None


def fit_forward_mtrf(stim, resp, fs, tmin_ms, tmax_ms, regularization, k):
    trf = TRF(direction=1)
    trf.train(stim, resp, fs, tmin_ms / 1e3, tmax_ms / 1e3, regularization, k = k)
    return trf.weights, trf.bias

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
## experiment module end
    
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
        self.weight = torch.nn.Parameter(torch.ones(outDim,inDim,nWin))
        self.bias = torch.nn.Parameter(torch.ones(outDim))
        k = 1 / (inDim * nWin)
        lower = - np.sqrt(k)
        upper = np.sqrt(k)
        torch.nn.init.uniform_(self.weight, a = lower, b = upper)
        torch.nn.init.uniform_(self.bias, a = lower, b = upper)
        self.ifAddBiasInForward = ifAddBiasInForward

    @property
    def outDim(self):
        return self.weight.shape[0]

    @property
    def inDim(self):
        return self.weight.shape[1]
    
    @property
    def nWin(self):
        return self.weight.shape[2]

    def forward(self,x):
        # x: (nBatch, inDim, nSeq)
        assert x.ndim == 3
        kernelsTemp =  self.weight[None, ..., None] #(1, outDim, inDim, nWin, 1) 
        xTemp = x[:, None, :, None, :] #(nBatch, 1, inDim, 1, nSeq)
        TRFs = xTemp * kernelsTemp  #(nBatch, outDim, inDim, nWin, nSeq)
        if self.ifAddBiasInForward:
            TRFs = TRFs + self.bias[..., None, None, None] #(nBatch, outDim, inDim, nWin, nSeq)
        TRFs = TRFs.sum(2) #(nBatch, outDim, nWin, nSeq)
        return TRFs

    def load_mtrf_weights(self, w, b, fs, device):
        #w: (nInChan, nLag, nOutChan)
        b = b[0]
        w = w * 1 / fs
        b = b * 1/ fs
        w = torch.from_numpy(w).to(device)
        b = torch.from_numpy(b).to(device)
        w = w.permute(2, 0, 1) #(nOutChan, nInChan, nLag)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(w)
            self.bias = torch.nn.Parameter(b)
        return self

    def export_mtrf_weights(self, fs):
        with torch.no_grad():
            # (nInChan, nLag, nOutChan)
            w = self.weight.cpu().detach().permute(1, 2, 0).numpy()
            b = self.bias.cpu().detach().numpy().reshape(1,-1)
        w = w * fs 
        b = b * fs
        return w, b
    
    def stop_update_weights(self):
        self.weight.requires_grad_(False)
        self.weight.grad = None
        self.bias.requires_grad_(False)
        self.bias.grad = None
        
    def enable_update_weights(self):
        self.requires_grad_(True)
        self.weight.grad = torch.zeros_like(
            self.weight
        )
        self.bias.grad = torch.zeros_like(
            self.bias
        )

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


class CustomKernelCNNTRF(torch.nn.Module):

    def __init__(self,inDim,outDim,tmin_ms,tmax_ms,fs,groups = 1,dilation = 1):
        super().__init__()
        self.tmin_ms = tmin_ms
        self.tmax_ms = tmax_ms
        self.fs = fs
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        self.tmin_idx = self.lagIdxs[0]
        self.tmax_idx = self.lagIdxs[-1]
        nLags = len(self.lagTimes)
        nKernels = (nLags - 1) / dilation + 1
        assert np.ceil(nKernels) == np.floor(nKernels)
        self.nWin = int(nKernels)
        self.oPadOrCrop = CPadOrCrop1D(self.tmin_idx,self.tmax_idx)
        self.groups = groups
        self.dilation = dilation
        self.inDim = inDim
        self.outDim = outDim

        k = 1 / (inDim * self.nWin)
        lower = - np.sqrt(k)
        upper = np.sqrt(k)
        self.bias = torch.nn.Parameter(torch.ones(outDim))
        torch.nn.init.uniform_(self.bias, a = lower, b = upper)

    def setTRFGen(self, trfGen):
        self.trfGen = trfGen

    def forward(self, x, weight = None):
        if weight is None:
            weight = self.trfGen.TRF()
        assert self.nWin == weight.shape[2]
        assert self.outDim == weight.shape[0]
        assert self.inDim == weight.shape[1]
        x = self.oPadOrCrop(x)
        y = torch.nn.functional.conv1d(
            x, 
            weight, 
            bias=self.bias, 
            dilation=self.dilation, 
            groups=self.groups
        )
        return y

class FuncBasisTRF(torch.nn.Module):

    def __init__(self, inDim, outDim, tmin_idx, tmax_idx, timeshiftLimit_idx ,device) -> None:
        super().__init__()
        self.timeshiftLimit_idx = torch.tensor(timeshiftLimit_idx)
        self.time_embedding = self.get_time_embedding(
            tmin_idx, tmax_idx, device) 
        self.time_embedding_ext = self.get_time_embedding(
            tmin_idx-timeshiftLimit_idx, tmax_idx+timeshiftLimit_idx, device) 
        nWin = self.time_embedding_ext.shape[-2]
        TRFs = torch.zeros((outDim, inDim, nWin),device=device)
        self.register_buffer('TRFs',TRFs)

    # def corrected_time_embedding(self, t):
        # return t + + self.timeshiftLimit_idx

    def TRF(self):
        # (outDim, inDim, nWin)
        #(nBatch, 1, 1, nWin, nSeq)
        return self.forward(1, 0, 1)[0,...,0]#.detach().cpu().numpy()      

    @property
    def inDim(self):
        return self.TRFs.shape[1]
    
    @property
    def outDim(self):
        return self.TRFs.shape[0]
    
    @property
    def nWin(self):
        return self.TRFs.shape[2]
    
    @property
    def nBasis(self):
        raise NotImplementedError
    
    def vis(self):
        raise NotImplementedError
    
    def forward(self, a,  b, c):
        raise NotImplementedError
    
    def fitTRFs(self, TRFs):
        raise NotImplementedError
    
    def get_time_embedding(self, tmin_idx, tmax_idx, device = 'cpu'):
        #(1, 1, 1, nWin, 1) 
        return torch.arange(tmin_idx,tmax_idx+1, device=device)\
            .view(1,1,1,-1,1)
    
    @property
    def timelag_idx(self,):
        return self.time_embedding.detach().cpu().squeeze()

    @property
    def timelag_idx_ext(self,):
        return self.time_embedding_ext.detach().cpu().squeeze()


def build_gaussian_response(x, mu, sigma):
    # x: (nBatch, 1, 1, nWin, nSeq)
    # mu: (nBasis)
    # sigma: (nBasis, outDim, inDim)
    # output: (nBatch, nBasis, outDim, inDim, nWin, nSeq)

    # x: (nBatch, 1, 1, 1, nWin, nSeq)
    if x.ndim == 5:
        x = x[:, None, ...]
    # mu: (nBasis, 1, 1, 1, 1)
    mu = mu[..., None, None, None, None]
    # sigma: (nBasis, outDim, inDim,  1, 1)
    sigma = sigma[..., None, None]
    # output: (nBatch, nBasis, outDim, inDim, nWin, nSeq)
    return torch.exp(-(x-mu)**2 / (2*(sigma)**2))

def solve_coef(gaussresps, trf):
    nWin = trf.shape[0]
    A = np.concatenate([gaussresps, np.ones((1,nWin))], axis = 0).T
    coefs = np.linalg.lstsq(A, trf, rcond=None)[0]
    return coefs

class GaussianBasisTRF(FuncBasisTRF):
    
    def __init__(
        self,
        inDim,
        outDim,
        tmin_idx, 
        tmax_idx,
        nBasis,
        timeshiftLimit_idx = 0,
        sigmaMin = 6.4,
        sigmaMax = 6.4,
        ifSumInDim = False,
        device = 'cpu',
        mu = None,
        sigma = None,
        include_constant_term = True
    ):
        super().__init__(inDim, outDim, tmin_idx, tmax_idx, timeshiftLimit_idx, device)
        nWin = self.nWin
        ### Fittable Parameters
        ## out projection init
        coefs = torch.ones((nBasis + 1, outDim, inDim), device = device, dtype = torch.float32)
        torch.nn.init.kaiming_uniform_(coefs, a=math.sqrt(5))
        self.coefs = torch.nn.Parameter(coefs)
        ## bias init
        # k = 1 / (inDim * nWin)
        # lower = - np.sqrt(k)
        # upper = np.sqrt(k)
        # self.bias = torch.nn.Parameter(torch.ones(outDim))
        # torch.nn.init.uniform_(self.bias, a = lower, b = upper)
        ## sigma init
        if sigma is not None:
            assert len(sigma) == nBasis
            sigma = torch.tensor(sigma)
        else:
            sigma = torch.ones(nBasis, outDim, inDim, device = device, dtype = torch.float32) * (sigmaMin + sigmaMax) / 2
        self.sigma = torch.nn.Parameter(sigma)
        # torch.nn.init.uniform_(self.sigma, a = lower, b = upper)

        ### Fixed Values
        # timeEmbed = torch.arange(nWin) 
        # self.register_buffer('timeEmbed', timeEmbed)
        time_embedding_ext = self.time_embedding_ext.squeeze()
        tmin_idx_ext, tmax_idx_ext = time_embedding_ext[0], time_embedding_ext[-1]
        if mu is not None:
            assert len(mu) == nBasis
            mu = torch.tensor(mu, device = device, dtype = torch.float32)
        else:
            mu = torch.linspace(tmin_idx_ext.item(), tmax_idx_ext.item(), nBasis + 2)[1:-1]
        self.register_buffer('mu', mu)
        # self.mu = torch.nn.Parameter(mu)

        sigmaMin = torch.tensor(sigmaMin)
        self.register_buffer('sigmaMin', sigmaMin)
        sigmaMax = torch.tensor(sigmaMax)
        self.register_buffer('sigmaMax', sigmaMax)
        self.ifSumInDim = ifSumInDim
        self.include_constant_term = include_constant_term
        self.device = device

    def vec_sum(self, x):
        return self.vec_gauss_sum(x)

    def vec_gauss_sum(self, x):
        sigma = self.sigma
        sigma = torch.maximum(sigma, self.sigmaMin)
        sigma = torch.minimum(sigma, self.sigmaMax)
        # print(sigma)
        # (nBatch, nBasis, outDim, inDim, nWin, nSeq)
        # print(self.mu, sigma)
        gaussResps = build_gaussian_response(
            x, 
            self.mu, 
            sigma
        )
        
        # coefs: (nBasis + 1, outDim, inDim, 1, 1)
        coefs = self.coefs[..., None, None]
        # print(gaussResps.shape)
        # nBatch, _, outDim, inDim, nWin, nSeq = gaussResps.shape
        # (nBatch, nBasis+1, outDim, inDim, nWin, nSeq)
        # aug_gaussResps = torch.cat([gaussResps, torch.ones(nBatch, 1, outDim, inDim, nWin, nSeq, device = self.device)], dim = -5)
        # # wGaussResps = coefs[:-1,...] * gaussResps
        # wGaussResps = coefs * aug_gaussResps
        # # (nBatch, outDim, inDim, nWin, nSeq)
        # wGaussResps = wGaussResps.sum((-5))

        # (nBasis, outDim, inDim, 1, 1)
        coefs_1 = coefs[:-1]
        # (outDim, inDim, 1, 1)
        coefs_2 = coefs[-1]
        # (nBatch, nBasis+1, outDim, inDim, nWin, nSeq)
        w_gauss_resps_1 = coefs_1 * gaussResps
        if self.include_constant_term:
            # print('include constant term')
            w_gauss_resps = w_gauss_resps_1.sum(-5) +  coefs_2
        else:
            w_gauss_resps = w_gauss_resps_1.sum(-5)

        return w_gauss_resps

    def forward(self, a, b, c):
        # print(x.shape)
        # output (nBatch, outDim, (inDim), nWin, nSeq)
        # x: x: (nBatch, 1, 1, nWin, nSeq)
        # currently just support the 'component' mode
        x = c * (self.time_embedding - b)
        # x = self.corrected_time_embedding(x)
        wGaussResps = a * self.vec_gauss_sum(x)
        # print(coefs[:,0,0,0,0])
        if self.ifSumInDim:
            # (nBatch, outDim, nWin, nSeq)
            wGaussResps = wGaussResps.sum((-3))
            # wGaussResps = wGaussResps + self.bias[:, None, None]
        # print(wGaussResps)
        return wGaussResps
    
    @property
    def nBasis(self):
        return self.sigma.shape[0]
    
    def fitTRFs(self, TRFs):
        '''
        TRFs is the numpy array of mtrf weights
        Shape: [nInDim, nLags, nOutput]

        self.coefs: (nBasis+1, outDim, inDim)
        sigma: (nBasis, outDim, inDim)
        '''
        # print(TRFs.shape)
        TRFs = torch.from_numpy(TRFs)
        TRFs = TRFs.permute(2, 0, 1)
        self.TRFs[:,:,:] = TRFs.to(self.device)[:,:,:]
        x = self.time_embedding_ext
        sigma = self.sigma
        # print(sigma)
        # (nBasis, outDim, inDim, nWin)
        with torch.no_grad():
            gaussResps = build_gaussian_response(
                x, 
                self.mu, 
                sigma
            )[0, ..., 0].cpu().numpy()

        nWin = TRFs.shape[2]
        assert nWin == self.nWin
        # (nBasis+1, outDim, inDim)
        coefs = np.zeros(self.coefs.shape)
        for i in range(self.outDim):
            for j in range(self.inDim):
                t_trf = TRFs[i,j,:]
                # (nBasis, nWin)
                t_gauss = gaussResps[:, i, j, :] 
                t_coef = solve_coef(
                    t_gauss,
                    t_trf
                )
                # print(coefs[:, i, j].shape, t_coef.shape)
                coefs[:, i, j] = t_coef
        
        with torch.no_grad():
            self.coefs[:,:,:] = torch.from_numpy(coefs)
            # (nBatch, outDim, inDim, nWin, nSeq)
            torchTRFs = self.vec_gauss_sum(
                self.time_embedding_ext,
            ).cpu().numpy()[0, ..., 0]
            for j in range(self.outDim):
                for i in range(self.inDim):
                    curFTRF = torchTRFs[j, i,:]
                    TRF = TRFs[j, i, :]
                    # print(pearsonr(curFTRF, TRF))
                    assert np.around(pearsonr(curFTRF, TRF)[0]) >= 0.99

    def vis(self, fs = None):
        if plt is None:
            raise ValueError('matplotlib should be installed')
        with torch.no_grad():
            FTRFs = self.vec_gauss_sum(
                self.time_embedding_ext,
            ).cpu()[0, ..., 0]
        nInChan = self.inDim
        nOutChan = self.outDim
        fig, axs = plt.subplots(2)
        fig.suptitle('top: original TRF, bottom: reconstructed TRF')
        if fs is None:
            timelag = self.timelag_idx_ext
        else:
            timelag = self.timelag_idx_ext.numpy() / fs
        for j in range(nOutChan):
            for i in range(nInChan):
                TRF = self.TRFs[j,i,:].cpu()
                FTRF = FTRFs[j,i,:].cpu()
                # print(pearsonr(FTRF, TRF)[0])
                axs[0].plot(timelag, TRF)
                # if j == 0:
                axs[1].plot(timelag, FTRF)
        return fig



class FourierBasisTRF(FuncBasisTRF):
    
    def __init__(
        self,
        nInChan,
        nOutChan,
        tmin_idx,
        tmax_idx,
        nBasis,
        timeshiftLimit_idx,
        device = 'cpu',
        if_fit_coefs = False
    ):
        #TRFs the TRF for some channels
        super().__init__(nInChan, nOutChan, tmin_idx, tmax_idx, timeshiftLimit_idx, device)
        # self.nBasis = nBasis
        # self.nInChan = nInChan
        # self.nOutChan = nOutChan
        # self.nWin = nWin
        coefs = torch.empty((nOutChan, nInChan, nBasis),device=device)
        if if_fit_coefs:
            torch.nn.init.kaiming_uniform_(coefs, a=math.sqrt(5))
            self.coefs = torch.nn.Parameter(coefs)
        else:
            self.register_buffer('coefs', coefs)
        self.T = self.nWin - 1
        self.device = device
        maxN = nBasis // 2
        self.seqN = torch.arange(1,maxN+1,device = self.device)
        # self.saveMem = False #expr for saving memory usage

    @property
    def nBasis(self):
        return self.coefs.shape[2]

    def fitTRFs(self,TRFs):
        '''
        TRFs is the numpy array of mtrf weights
        Shape: [nInDim, nLags, nOutput]
        '''

        TRFs = torch.from_numpy(TRFs)
        TRFs = TRFs.permute(2, 0, 1)
        self.TRFs[:,:,:] = TRFs.to(self.device)[:,:,:]
        fd_basis_s = []
        # grid_points = list(range(self.nWin))
        grid_points = self.time_embedding_ext.squeeze().cpu().numpy()
        for j in range(self.outDim):
            for i in range(self.inDim):
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
                
        out = self.vec_fourier_sum(
            self.nBasis,
            self.T,
            self.time_embedding_ext,
            self.coefs
        )[0, ..., 0]
        for j in range(self.outDim):
            for i in range(self.inDim):
                fd_basis = fd_basis_s[j*self.inDim + i]
                temp = fd_basis(grid_points).squeeze() #np.arange(0,self.nWin)
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
    
    def vec_sum(self, x):
        return self.vec_fourier_sum(self.nBasis, self.T, x, self.coefs)
    
    def vec_fourier_sum(self,nBasis, T, t,coefs):
        #coefs: (nOutChan, nInChan, nBasis)
        #t: (nBatch, 1, 1, nWin, nSeq)
        #   if tChan of t is just 1, which means we share
        #   the same time-axis transformation for all channels
        #return: (nBatch, outDim, inDim, nWin, nSeq)

        #(nBatch, 1, 1, nWin, nSeq, 1)
        t = t[..., None]
        # print(t.shape)
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

        nBatch, _, _, nWin, nSeq, nBasis =  constN.shape
        nOutChan, nInChan, nBasis = coefs.shape

        #(nOutChan, nInChan, 1, 1, nBasis)
        coefs = coefs[:, :, None, None, :]
        nBasis = nBasis
        # print(constN.shape, coefs.shape)
        '''
        #expr for saving memory usage
        memAvai,_ = torch.cuda.mem_get_info()
        nMemReq = nBatch * nSeq * nInChan * nOutChan * nBasis * nWin * 4 # 4 indicates 4 bytes
        # print(torch.cuda.memory_allocated()/1024/1024)
        if nMemReq > memAvai * 0.9 or self.saveMem:
            out = const0 * coefs[...,0] #(nOutChan, nInChan, 1, 1)
            for nB in range(2 * maxN):
                out = out + constN[...,nB] * coefs[...,1+nB]
        else:
            # (nbatch, nOutChan, nInChan, nWin, nSeq, nBasis)
            out =  const0 * coefs[...,0] + (constN * coefs[...,1:]).sum(-1)
        '''   
        out =  const0 * coefs[...,0] + (constN * coefs[...,1:]).sum(-1)
        # print(torch.cuda.memory_allocated()/1024/1024)
        # (nBatch, outDim, inDim, nWin, nSeq)
        return out
    
    def forward(self,a, b, c):
        # a,b,c in most strict case (nBatch, 1, 1, 1, nSeq)
        # loosly: a,b,c can be (nBatch, nOut, nIn, nWin, nSeq)

        #self.time_embedding #(1, 1, 1, nWin, 1) 
        
        # x: (nBatch, 1, 1, nWin, nSeq) 
        nSeq = self.time_embedding
        x = c * (nSeq - b)
        # x = self.corrected_time_embedding(x)
        #(nBatch, outDim, inDim, nWin, nSeq)
        # nonLinTRFs = aSeq * self.basisTRF( cSeq * ( nSeq -  bSeq) ) 

        # return: 
        coefs = self.coefs
        out = a * self.vec_fourier_sum(self.nBasis,self.T,x,coefs)
        return out
        
    def vis(self ,fs = None):
        if fs is None:
            timelag = self.timelag_idx_ext
        else:
            timelag = self.timelag_idx_ext.numpy() / fs
        if plt is None:
            raise ValueError('matplotlib should be installed')
        with torch.no_grad():
            FTRFs = self.vec_fourier_sum(
                self.nBasis,
                self.T,
                self.time_embedding_ext,
                self.coefs
            )[0, ..., 0]
        nInChan = self.inDim
        nOutChan = self.outDim
        fig, axs = plt.subplots(2)
        fig.suptitle('top: original TRF, bottom: reconstructed TRF')
        for j in range(nOutChan):
            for i in range(nInChan):
                TRF = self.TRFs[j,i,:].cpu()
                FTRF = FTRFs[j,i,:].cpu()
                axs[0].plot(timelag, TRF)
                axs[1].plot(timelag, FTRF)
        return fig

basisTRFNameMap = {
    'gauss': GaussianBasisTRF,
    'fourier': FourierBasisTRF
}


class TRFsGen(torch.nn.Module):

    def forward(self, x, featOnsetIdx):
        '''
        input:
             x: input to be used to derive the 
                transformation parameters TRFs
             featOnsetIdx: time index of item 
                in the transformed x to be picked as real transformation parameters
        '''
        pass


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
        basisTRFName = 'fourier',
        limitOfShift_idx = 7,
        nBasis = 21, 
        mode = '',
        transformer = None,
        device = 'cpu',
        # if_trans_per_outChan = False
    ):
        super().__init__()
        assert mode.replace('+-','') in ['','a','b','a,b','a,b,c','a,c']
        self.fs = fs
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagIdxs_ts = torch.Tensor(self.lagIdxs).float().to(device)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        nWin = len(self.lagTimes)
        self.mode = mode
        self.transformer:torch.nn.Module = transformer
        self.n_transform_params = len(mode.split(','))
        self.device = device
        if isinstance(basisTRFName, str):
            self.basisTRF:FuncBasisTRF = basisTRFNameMap[basisTRFName](
                inDim, 
                outDim, 
                self.lagIdxs[0],
                self.lagIdxs[-1], 
                nBasis,
                timeshiftLimit_idx = limitOfShift_idx, 
                device=device
            )
        elif isinstance(basisTRFName, torch.nn.Module):
            self.basisTRF:FuncBasisTRF = basisTRFName
        else:
            raise ValueError()
        # self.if_trans_per_outChan = if_trans_per_outChan

        if transformer is None:
            transInDim, transOutDim, device = self.get_default_transformer_param()
            self.transformer:torch.nn.Module = CausalConv(transInDim, transOutDim, 2).to(device)

        self.limitOfShift_idx = torch.tensor(limitOfShift_idx)

    @classmethod
    def parse_trans_params(cls,mode):
        return mode.split(',')

    @property
    def tmin_ms(self):
        return self.lagTimes[0]
    
    @property
    def tmax_ms(self):
        return self.lagTimes[-1]

    @property
    def inDim(self):
        return self.basisTRF.inDim
    
    @property
    def outDim(self):
        return self.basisTRF.outDim
    
    @property
    def nWin(self):
        return self.basisTRF.nWin - 2 * self.limitOfShift_idx
    
    @property
    def nBasis(self):
        return self.basisTRF.nBasis

    @property
    def extendedTimeLagRange(self):
        minLagIdx = self.lagIdxs[0]
        maxLagIdx = self.lagIdxs[-1]
        left = np.arange(minLagIdx - self.limitOfShift_idx, minLagIdx)
        right = np.arange(maxLagIdx + 1, maxLagIdx + 1 + self.limitOfShift_idx)
        extLag_idx = np.concatenate([left, self.lagIdxs, right])
        try:
            assert len(extLag_idx) == self.nWin + 2 * self.limitOfShift_idx
        except:
            print(len(extLag_idx), self.nWin + 2 * self.limitOfShift_idx)
        timelags = Idxs2msec(extLag_idx, self.fs)
        return timelags[0], timelags[-1]

    def get_default_transformer_param(self):
        inDim = self.inDim
        device = self.device
        outDim = self.n_transform_params
        return inDim, outDim, device

    def fitFuncTRF(self, w):
        # do the self.fs operation here because basisTRF doesn't keep fs info
        w = w * 1 / self.fs 
        with torch.no_grad():
            self.basisTRF.fitTRFs(w)
        return self
    
    def pickParam(self,paramSeqs,idx):
        #paramSeqs: (nBatch, nMiddleParam, nOut/1, 1, 1, nSeq)
        return paramSeqs[:, idx, ...]
    
    
    def getTransformParams(self, x, startIdx = None):
        paramSeqs = self.transformer(x) #(nBatch, nMiddleParam, nSeq)
        nBatch, nMiddleParam, nSeq = paramSeqs.shape
        if startIdx is not None:
            idxBatch = torch.arange(nBatch)
            idxMiddleParam = torch.arange(nMiddleParam)
            idxMiddleParam = idxMiddleParam[:, None]
            startIdx = startIdx[:, None, :]
            idxBatch = idxBatch[:, None, None]
            paramSeqs = paramSeqs[idxBatch, idxMiddleParam, startIdx]
        
        #(nBatch, n_transform_params, 1, 1, nSeq), this is the most strict case
        #however, it can also be (nBatch, n_transform_params, nOut, 1, nSeq) 
        #  for different transformation for different channel
            
        paramSeqs = paramSeqs.view(nBatch, self.n_transform_params, -1, 1, 1, nSeq) #[:, :, None, None, :]
        if paramSeqs.shape[2] != 1:
            assert paramSeqs.shape[2] == self.outDim
        midParamList = self.mode.split(',')
        if midParamList == ['']:
            midParamList = []
        nParamMiss = 0
        if 'a' in midParamList:
            aIdx = midParamList.index('a')
            #(nBatch, 1, nOut/1, 1, nSeq)
            aSeq = self.pickParam(paramSeqs, aIdx) 
            aSeq = torch.abs(aSeq)
        elif '+-a' in midParamList:
            aIdx = midParamList.index('+-a')
            #(nBatch, 1, nOut/1, 1, nSeq)
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
            # print('c: ',cSeq.squeeze())
            #two reasons, cSeq must be larger than 0; 
            #if 1 is the optimum, abs will have two x for the optimum, 
            # which is not stable 
            # cSeq = torch.tanh(cSeq) #expr
            cSeq =  1 + cSeq
            cSeq = torch.maximum(cSeq, torch.tensor(0.5))
            cSeq = torch.minimum(cSeq, torch.tensor(1.28))
        else:
            nParamMiss += 1
            cSeq = 1

        assert (len(midParamList) + nParamMiss) == 3
        return aSeq, bSeq, cSeq

    def forward(self, x, featOnsetIdx = None):
        '''
        x: (nBatch, inDim, nSeq)
        output: TRFs (nBatch, outDim, nWin, nSeq)
        '''
        #(nBatch, nOut/1, 1, 1, nSeq)
        aSeq, bSeq, cSeq = self.getTransformParams(x, featOnsetIdx)
        #(1, 1, 1, nWin, 1) 
        # nSeq = self.lagIdxs_ts[None, None, None, :, None] + self.limitOfShift_idx 
        # print(aSeq, bSeq, cSeq)x
        #(nBatch, outDim, inDim, nWin, nSeq)
        # nonLinTRFs = aSeq * self.basisTRF( cSeq * ( nSeq -  bSeq) ) 
        # print(aSeq.shape, bSeq.shape)
        nonLinTRFs = self.basisTRF(aSeq, bSeq, cSeq)
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


    the core mechanism of this module following thess steps:
        1. generate TRFs using the input param 'x',
        2. determine the onset time of these TRFs within the output time series, using input param 'timeinfo',
        3. align and sum the generated TRFs at their corresponding time location in the output.
    
    Note:
        1. currently, there are two types of x supported, 
            type 1: discrete type, means there is a single x for each time point in the 'timestamp'
            type 2: continuous type, means there is a timeseries of x which has the same length as the output timeseries,
                and when generating TRFs, only part of the x will actually contribute to this process.

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
        x_is_timeseries = False,
        verbose = True
    ):
        '''
        inDim: int, the number of columns of input 
        outDim: int, the number of columns of output of ltiTRFGen and trfsGen

        '''
        super().__init__()
        assert tmin_ms >= 0
        self.x_is_timeseries = x_is_timeseries
        self.lagIdxs = msec2Idxs([tmin_ms,tmax_ms],fs)
        self.lagTimes = Idxs2msec(self.lagIdxs,fs)
        nWin = len(self.lagTimes)
        self.ltiTRFsGen = LTITRFGen(
            inDim,
            nWin,
            outDim,
            ifAddBiasInForward=False
        ).to(device)
        # if callable(trfsGen):
        #     trfsGen = trfsGen()
        self.trfsGen:FuncTRFsGen = trfsGen if trfsGen is None else trfsGen.to(device)
        self.fs = fs

        self.bias = None
        #also train bias for the trfsGen provided by the user
        if self.trfsGen is not None:
            self.init_nonLinTRFs_bias(inDim, nWin, outDim, device)
        
        self.trfAligner = TRFAligner(device)
        self._enableUserTRFGen = True 
        self.device = device
        self.verbose = verbose

    @property
    def inDim(self):
        return self.ltiTRFsGen.inDim
    
    @property
    def outDim(self):
        return self.ltiTRFsGen.outDim
    
    @property
    def nWin(self):
        return self.ltiTRFsGen.nWin
    
    @property
    def tmin_ms(self):
        return self.lagTimes[0]
    
    @property
    def tmax_ms(self):
        return self.lagTimes[-1]
    
    def init_nonLinTRFs_bias(self, inDim, nWin, outDim, device):
        self.bias = torch.nn.Parameter(torch.ones(outDim))
        fan_in = inDim * nWin
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def set_trfs_gen(self, trfsGen):
        self.trfsGen = trfsGen.to(self.device)
        self.bias = torch.nn.Parameter(
            torch.ones(self.outDim, device = self.device)
        )
        fan_in = self.inDim * self.nWin
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def get_params_for_train(self):
        return [i for i in self.trfsGen.transformer.parameters()] + [self.bias]
        # raise NotImplementedError()
        

    def set_linear_weights(self, w, b):
        #w: (nInChan, nLag, nOutChan)
        self.ltiTRFsGen.load_mtrf_weights(
            w,
            b,
            self.fs,
            self.device
        )
        return self

    def get_linear_weights(self):
        w, b = self.ltiTRFsGen.export_mtrf_weights(
            self.fs
        )
        return w, b

    @property
    def if_enable_trfsGen(self):
        return self._enableUserTRFGen

    @if_enable_trfsGen.setter
    def if_enable_trfsGen(self,x):
        assert isinstance(x, bool)
        if self.verbose:
            print('set ifEnableNonLin',x)
        if x == True and self.trfsGen is None:
            raise ValueError('trfGen is None, cannot be enabled')
        self._enableUserTRFGen = x

    def stop_update_linear(self):
        self.ltiTRFsGen.stop_update_weights()
        
    def enable_update_linear(self):
        self.ltiTRFsGen.enable_update_weights()

    def forward(self, x, timeinfo):
        '''
        input: 
            x: nBatch * [nChan, nSeq], 
            timeinfo: nBatch * [2, nSeq]
        output: targetTensor
        '''

        ### record the necessary information of each item in the batch
        ### for x and targetTensor
        nSeqs = [] # length of x in the batch
        nRealLens = [] # length of real output in the batch
        trfOnsetIdxs = [] # the corresponding index of timepoint in the timeinfo in the batch
        for ix, xi in enumerate(x):
            nLenXi = xi.shape[-1]
            if timeinfo[ix] is not None:
                # print(timeinfo[ix].shape)
                if not self.x_is_timeseries:
                    assert timeinfo[ix].shape[-1] == xi.shape[-1], f"{timeinfo[ix].shape[-1]} != {xi.shape[-1]}"
                nLen = torch.ceil(
                    timeinfo[ix][0][-1] * self.fs
                ).long() + self.nWin
                onsetIdx = torch.round(
                    timeinfo[ix][0,:] * self.fs
                ).long() + self.lagIdxs[0]
            else:
                nLen = nLenXi
                onsetIdx = torch.tensor(np.arange(nLen)) + self.lagIdxs[0]
            nSeqs.append(nLenXi)
            nRealLens.append(nLen)
            trfOnsetIdxs.append(onsetIdx)

        nGlobLen = max(nRealLens)
        x = seqLast_pad_zero(x)
        trfOnsetIdxs = seqLast_pad_zero(trfOnsetIdxs, value = -1)
        
        # if x is time series
        featOnsetIdxs = None
        if self.x_is_timeseries:
            # featIdxs is the index where we get the feat for generating TRFs
            featOnsetIdxs = trfOnsetIdxs.detach().clone()
            featOnsetIdxs[featOnsetIdxs != -1] =\
                featOnsetIdxs[featOnsetIdxs != -1] - self.lagIdxs[0]
        
        #TRFs shape: (nBatch, outDim, nWin, nSeq)
        # print(x.shape)
        TRFs = self.get_trfs(x, featOnsetIdxs)

        #targetTensor shape: (nBatch,outDim,nRealLen)
        targetTensor = self.trfAligner(TRFs,trfOnsetIdxs,nGlobLen)

        if self.if_enable_trfsGen:
            targetTensor = targetTensor + self.bias.view(-1,1)
        else:
            ltiTRFBias = self.ltiTRFsGen.bias
            targetTensor = targetTensor + ltiTRFBias.view(-1,1)

        return targetTensor
    
    def get_trfs(self, x, featOnsetIdxs = None):
        if self.if_enable_trfsGen:
            return self.trfsGen(x, featOnsetIdxs)
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

