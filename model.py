import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import math



class Interpolation(nn.Module):
    def __init__(self, size=None, scale_factor=(2.0, 2.0), mode='bilinear'):
        super(Interpolation, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return F.interpolate(input, size=self.size, scale_factor=self.scale_factor, mode=self.mode)

class Generater(nn.Module):
    def __init__(self, opt):
        super(Generater, self).__init__()
        self.encoder_decoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.ngf, 4, 2, 1, bias=False),
            nn.ELU(alpha=3, inplace=True),
            # state size: (nef) x 64 x 64
            nn.Conv2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ELU(alpha=3, inplace=True),
            # state size: (nef) x 32 x 32
            nn.Conv2d(opt.ngf, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ELU(alpha=3, inplace=True),
            # state size: (nef*2) x 16 x 16
            nn.Conv2d(opt.ngf * 2, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ELU(alpha=3, inplace=True),
            # state size: (nef*4) x 8 x 8
            nn.Conv2d(opt.ngf * 4, opt.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ELU(alpha=3, inplace=True),
            # state size: (nef*8) x 4 x 4
            nn.Conv2d(opt.ngf * 8, opt.nBottleneck, 4, bias=False),
            # state size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(opt.nBottleneck),
            nn.ELU(alpha=3, inplace=True),

            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ELU(alpha=3, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ELU(alpha=3, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ELU(alpha=3, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ELU(alpha=3, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(opt.ngf, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ELU(alpha=3, inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        x = input
        x = self.encoder_decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)



