import torch

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

class ASTRF(torch.nn.Module):
    '''
    the TRF implemented the convolution sum of temporal response,
        (i.e., time-aligning the temporal responses at their 
        corresponding location, and point-wise sum them).
        It requres a module to generate temproal responses to each
        individual stimuli, and also require time information to
        displace/align the temporal responses at the right
        indices/location 
    '''

    def __init__(
        self,
        inDim,
        outDim,
        tmin_ms,
        tmax_ms,
        fs,
        device = 'cpu'
    ):
        super().__init__()
        pass

    def forward(self,):
        pass

    def oneOfBatch(self, x, timeinfo):
        TRFs = self.getTRFs(x, timeinfo)

    def getTRFs(self, x, timeinfo):
        pass

