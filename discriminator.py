import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ConvInNormLeakyReluLayer

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = ConvInNormLeakyReluLayer(in_channels=3, out_channels=64, doNorm=False)
        self.c2 = ConvInNormLeakyReluLayer(in_channels=64, out_channels=128)
        self.c3 = ConvInNormLeakyReluLayer(in_channels=128, out_channels=256)
        self.c4 = ConvInNormLeakyReluLayer(in_channels=256, out_channels=512, stride=1)
        self.c5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.c4(out)
        out = self.c5(out)
        return out

batch_size = 16
img_size = 128
dummy = torch.randn(batch_size, 3, img_size, img_size)
#r1 = ConvInNormLeakyReluLayer(in_channels=3, out_channels=64)
model = Discriminator()
out = model(dummy)
print(out.shape)