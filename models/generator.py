import torch.nn as nn
import torch.nn.functional as F
from models.layers import ConvInNormReluLayer, ResidualLayer, UpConvInNormReluLayer



class Generator(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.c7_64 = ConvInNormReluLayer(in_channels, out_channels=64, kernel_size=7, stride=1, padding="same")
        self.d1 = ConvInNormReluLayer(in_channels=64, out_channels=128)
        self.d2 = ConvInNormReluLayer(in_channels=128, out_channels=256)
        self.r1 = ResidualLayer(channels=256)
        self.r2 = ResidualLayer(channels=256)
        self.r3 = ResidualLayer(channels=256)
        self.r4 = ResidualLayer(channels=256)
        self.r5 = ResidualLayer(channels=256)
        self.r6 = ResidualLayer(channels=256)
        self.u1 = UpConvInNormReluLayer(in_channels=256, out_channels=128)
        self.u2 = UpConvInNormReluLayer(in_channels=128, out_channels=64)
        self.c7_3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding="same", padding_mode="reflect")

    def forward(self, x):
        
        out = self.c7_64(x)
        out = self.d1(out)
        out = self.d2(out)
        out = self.r1(out)
        out = self.r2(out)
        out = self.r3(out)
        out = self.r4(out)
        out = self.r5(out)
        out = self.r6(out)
        out = self.u1(out)
        out = self.u2(out)
        out = self.c7_3(out)
        out = F.tanh(out)
        return out

