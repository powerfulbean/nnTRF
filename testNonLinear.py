import torch
from nntrf.models import ASTRF, FuncTRFsGen
device = torch.device('cuda')
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