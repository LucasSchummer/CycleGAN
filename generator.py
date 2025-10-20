import torch
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

# batch_size = 16
# img_size = 128

# dummy = torch.randn(batch_size, 3, img_size, img_size)
# r1 = ConvInNormReluLayer(in_channels=3, out_channels=64)
# c1 = ConvInNormReluLayer(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding="same")
# out = c1(dummy)

# dummy = torch.randn(batch_size, 64, 32, 32)
# u1 = UpConvInNormReluLayer(64, 64)
# out = u1(dummy)

# model = Generator()
# out = model(dummy)

# print(out.shape)
