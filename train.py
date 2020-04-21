import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm.autonotebook import tqdm
from model import *
from defect import DefectAdder


class Config(object):
    data_path = r'C:\Users\PC\Pictures\test'
    num_workers = 0
    image_size = 96
    batch_size = 2
    max_epoch = 30
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
    netd_path = None
    netg_path = None

    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1


opt = Config()

transforms = tv.transforms.Compose([
    tv.transforms.Resize(opt.image_size),
    tv.transforms.CenterCrop(opt.image_size),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)

map_location = lambda storage, loc: storage
netd = Discriminator(opt)
netg = Generater(opt)

if opt.netd_path:
    netd.load_state_dict(torch.load(opt.netd_path, map_location=map_location))
if opt.netg_path:
    netg.load_state_dict(torch.load(opt.netg_path, map_location=map_location))

optimizer_g = optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))

criterion = nn.BCELoss()
contrast_criterion = nn.L1Loss(reduction='mean')

defect_adder = DefectAdder()

true_labels = torch.ones(opt.batch_size)
fake_labels = torch.zeros(opt.batch_size)
# fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1)
# noises = torch.randn(opt.batch_size, opt.nz, 1, 1)

if opt.use_gpu:
    netd.cuda()
    netg.cuda()
    criterion.cuda()
    true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
    # fix_noises, noises = fix_noises.cuda(), noises.cuda()



for epoch in range(opt.max_epoch):
    progressbar = tqdm(dataloader)
    for ii, (img, _) in enumerate(progressbar):
        defect_img = defect_adder(img)
        real_img = torch.Tensor(img)
        if opt.use_gpu:
            real_img = real_img.cuda()

        if (ii + 1) % opt.d_every == 0:
            optimizer_d.zero_grad()

            output = netd(real_img)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            fake_img = netg(real_img).detach()
            fake_output = netd(fake_img)
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

        if (ii + 1) % opt.g_every == 0:
            optimizer_g.zero_grad()
            fake_img = netg(real_img)
            fake_output = netd(fake_img)

            error_g = criterion(fake_output, true_labels)
            error_g.backward()
            optimizer_g.step()
            progressbar.set_description(
                'Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}.'.format(
                    ii, error_d_real.item() + error_d_real.item(), error_g.item()))


