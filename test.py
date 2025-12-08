import torch
from layers import Lorentz_Conv2d
a = Lorentz_Conv2d(3, 16, 3, padding='same')

sp = torch.randn(8, 3, 32, 32)
time = ((sp ** 2).sum(dim=1, keepdim=True) + 1.0).sqrt()
x = torch.cat([time, sp], dim=1)

out = a(x)