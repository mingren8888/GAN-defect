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
        ngf = opt.ngf
        self.encoder = nn.Sequential(
            # nn.Conv2d(3, ngf, 5, 3, 1, bias=False),
            # nn.ReLU(),
            # stem/s2
            nn.Conv2d(3, ngf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # s4
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # s8
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # s16
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # s32
            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(),

            nn.Conv2d(ngf * 16, opt.nz, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1)
            # nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            #linear
            nn.Linear(opt.nz, math.ceil(opt.image_size/32)**2*16*ngf),

            nn.ConvTranspose2d(opt.nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        x = input
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input).view(-1)



