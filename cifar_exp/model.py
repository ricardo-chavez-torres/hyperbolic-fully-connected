from pathlib import Path
import sys
import torch

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers import Lorentz, Lorentz_Conv2d, Lorentz_fully_connected, LorentzBatchNorm


class ourModel(torch.nn.Module):
    def __init__(self, manifold: Lorentz):
        super(ourModel, self).__init__()

        self.manifold = manifold

        self.conv1 = Lorentz_Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        self.bn1 = LorentzBatchNorm(manifold, num_features=8)
        self.conv2 = Lorentz_Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        self.bn2 = LorentzBatchNorm(manifold, num_features=16)
        self.conv3 = Lorentz_Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        self.bn3 = LorentzBatchNorm(manifold, num_features=32)
        self.conv4 = Lorentz_Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding="same", bias=True, manifold=manifold)
        self.bn4 = LorentzBatchNorm(manifold, num_features=64)

        self.fc = Lorentz_fully_connected(in_features=64, out_features=100, manifold=manifold, do_mlr=True)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # (B, H*W, C)
        x = self.manifold.lorentz_midpoint(x)
        x = self.fc(x)
        return x
