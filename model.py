import torch
import torch.nn as nn
import torchvision


class Generater(nn.Module):
    def __init__(self, opt):
        super(Generater, self).__init__()
        ngf = opt.ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
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
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        ndf = opt.ndf
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()

        )

    def forward(self, *input):
        return self.main(input).view(-1)

class Config(object):

    data_path = 'data/'
    nuw_workers = 4
    image_size = 96
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4
    lr2 = 2e-4
    beta1 = 0.5
    use_gpu = True
    nz = 100
    ngf = 64
    ndf = 64

    save_path = 'imgs/'

    vis = True
    env = 'GAN'
    plot_every = 20

    debug_file = r'/tmp/debuggan'
    d_every = 1
    g_every = 5
    decay_every = 10
    netd_path = 'checkpoints/netd_211.pth'
    netg_path = 'checkpoints/netg_211.pth'

    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1

opt = Config()

