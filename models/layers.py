import torch.nn as nn
import torch.nn.functional as F


class ConvInNormReluLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode="reflect"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):

        out = self.conv(x)
        out = self.in1(out)
        out = F.relu(out)  
        return out
    

class ConvInNormLeakyReluLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect", doNorm=True):
        super().__init__()
        self.doNorm = doNorm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        if self.doNorm: self.in1 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        
        out = self.conv(x)
        if self.doNorm: out = self.in1(out)
        out = F.leaky_relu(out, .2)
        return out


class ResidualLayer(nn.Module):

    def __init__(self, channels, kernel_size=3, padding="same", padding_mode="reflect"):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
                
        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += x
        out = F.relu(out)
        return out
    

class UpConvInNormReluLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.in1 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        out = self.upconv(x)
        out = self.in1(out)
        out = F.relu(out)
        return out