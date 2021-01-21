import sys
import math
import torch
import torch.nn as nn

class Conv3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TransConv(nn.Module):
    def __init__(self, inputCh, outputCh):
        super(TransConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(inputCh, outputCh, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(outputCh),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Upsam(nn.Module):
    def __init__(self, inputCh, outputCh):
        super(Upsam, self).__init__()
        self.upconv = TransConv(inputCh, outputCh)#反卷积
        self.conv = Conv3x3(2 * outputCh, outputCh)#这里用到上面写的conv操作

    def forward(self, x, convfeatures):
        x = self.upconv(x)
        x = torch.cat([x, convfeatures], dim=1)
        x = self.conv(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=2, degree=64):
        super(UNet2D, self).__init__()

        chs = []
        for i in range(5):
            chs.append((2 ** i) * degree)
        self.downLayer1 = Conv3x3(in_ch, chs[0])
        self.downLayer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        Conv3x3(chs[0], chs[1]))

        self.downLayer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        Conv3x3(chs[1], chs[2]))

        self.downLayer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        Conv3x3(chs[2], chs[3]))

        self.bottomLayer = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                                        Conv3x3(chs[3], chs[4]))

        self.upLayer1 = Upsam(chs[4], chs[3])
        self.upLayer2 = Upsam(chs[3], chs[2])
        self.upLayer3 = Upsam(chs[2], chs[1])
        self.upLayer4 = Upsam(chs[1], chs[0])

        self.outLayer = nn.Conv2d(chs[0], out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x1 = self.downLayer1(x)
        x2 = self.downLayer2(x1)
        x3 = self.downLayer3(x2)
        x4 = self.downLayer4(x3)
        x5 = self.bottomLayer(x4)

        x = self.upLayer1(x5, x4)
        x = self.upLayer2(x, x3)
        x = self.upLayer3(x, x2)
        x = self.upLayer4(x, x1)
        x = self.outLayer(x)
        return x


if __name__ == "__main__":
    net = UNet2D(4, 5, degree=64)
    batch_size = 4
    a = torch.randn(batch_size, 4, 192, 192)
    b = net(a)
    print(b.shape)


