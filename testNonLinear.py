import torch
from nntrf.models import ASTRF
device = torch.device('cuda')
model = ASTRF(1, 128, 0, 700, 64, device = device)

x = torch.rand((1,1,10))
timeinfo = torch.tensor([
    [[1,3,5, 7,20, 40,45,60,75,90],
    [1,3,5, 7,20, 40,45,60,75,90]]]).float()
x = x.to(device)
timeinfo = timeinfo.to(device)

output = model(x, timeinfo)
print(output.shape)