import torch

class TwoMixedTRF(torch.nn.Module):
        
    def __init__(
        self,
        device,
        trfs, #list of trf models
        feats_keys, #list of feat key for each trf in the trfs
    ):
        super().__init__()
        self.trfs:torch.nn.Module = torch.nn.ModuleList([trf for trf in trfs])
        self.feats_keys = feats_keys
        self.device = device

    def forward(self,feat_dict:dict, y):

        pred_list = []
        for trf_index, iTRF in enumerate(self.trfs):
            feats_key = self.feats_keys[trf_index]
            feats = []
            n_dict_feat = 0
            for feat_key in feats_key:
                feat = feat_dict[feat_key]
                if isinstance(feat, dict):
                    feats.append(feat)
                    n_dict_feat += 1
                else:
                    assert isinstance(feat, torch.Tensor)
                    feats.append(feat)
            if n_dict_feat > 0:
                assert len(feats) == n_dict_feat
                # concatente
                if len(feats) == 1:
                    feats = feats[0]
                else:
                    # raise NotImplementedError
                    timeinfo_0 = feats[0]['timeinfo']
                    xs = []
                    for feat in feats:
                        xs.append(feat['x'])
                        torch.equal(timeinfo_0, feat['timeinfo'])
                    xs = torch.cat(xs, dim = -2)
                    feats = {
                        'x':xs,
                        'timeinfo':timeinfo_0
                    }
                pred_list.append(iTRF(**feats))
            else:
                minLen = min([f.shape[-1] for f in feats])
                feats = torch.cat([f[...,:minLen] for f in feats], axis = -2)
                pred_list.append(iTRF(feats))
            


        minLen = min([p.shape[-1] for p in predList] + [y.shape[-1]])
            
        predList = [p[...,:minLen] for p in predList]
        cropedY = y[:,:,:minLen]

        if len(predList) == 2:
            pred = predList[0] + predList[1]
        else:
            pred = predList[0]
        pred = sum(predList)
        # stop
        return pred,cropedY