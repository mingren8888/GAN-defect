import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm.autonotebook import tqdm
from model import *
from defect import DefectAdder, NormalizeList, ToTensorList
from utils import *
import os


class Config(object):
    data_path = r'/data/sdv2/GAN/data/gan_defect/GRID'
    save_path = '/data/sdv2/GAN/GAN_defect/imgs/0424_grid_1'
    work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0424_grid_1'

    num_workers = 4
    image_size = 96
    batch_size = 16
    max_epoch = 300
    steps = [200, 260]
    lrg = 1e-5
    lrd = 1e-5
    beta1 = 0.5
    use_gpu = True
    nz = 100
    ngf = 64
    ndf = 64

    contrast_loss_weight = 1



    vis = True
    env = 'GAN'
    plot_every = 20

    d_every = 1
    g_every = 1
    decay_every = 10
    netd_path = '/data/sdv2/GAN/GAN_defect/workdirs/0423/d_ckpt_e1000.pth'
    netg_path = '/data/sdv2/GAN/GAN_defect/workdirs/0423/g_ckpt_e1000.pth'
    # netd_path = None
    # netg_path = None

    gen_img = 'result.png'
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    checkpoint_interval = 100

    debug = True


opt = Config()

transforms = tv.transforms.Compose([
    tv.transforms.Resize(opt.image_size),
    tv.transforms.CenterCrop(opt.image_size),
    # tv.transforms.ToTensor(),
    DefectAdder(),
    ToTensorList(),
    NormalizeList(opt.mean, opt.std),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    print('loading checkpoint for discriminator...')
    netd.load_state_dict(torch.load(opt.netd_path, map_location=map_location)['net'])
if opt.netg_path:
    print('loading checkpoint for generator...')
    netg.load_state_dict(torch.load(opt.netg_path, map_location=map_location)['net'])

optimizer_g = optim.Adam(netg.parameters(), opt.lrg, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(netd.parameters(), opt.lrd, betas=(opt.beta1, 0.999))

scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)

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

if not os.path.exists(opt.work_dir):
    os.makedirs(opt.work_dir)

for epoch in range(opt.max_epoch):
    progressbar = tqdm(dataloader)
    d_loss = AverageMeter()
    g_loss = AverageMeter()
    c_loss = AverageMeter()
    if (epoch + 1) % opt.checkpoint_interval == 0:
        state_d = {'net': netd.state_dict(), 'optimizer': optimizer_d.state_dict(), 'epoch': epoch}
        state_g = {'net': netg.state_dict(), 'optimizer': optimizer_g.state_dict(), 'epoch': epoch}
        print('saving checkpoints...')
        torch.save(state_d, os.path.join(opt.work_dir, f'd_ckpt_e{epoch + 1}.pth'))
        torch.save(state_g, os.path.join(opt.work_dir, f'g_ckpt_e{epoch + 1}.pth'))
    for ii, (imgs, _) in enumerate(progressbar):
        normal, defect = imgs
        if opt.use_gpu:
            normal = normal.cuda()
            defect = defect.cuda()

        if (ii + 1) % opt.d_every == 0:
            # train discriminator
            netd.train()
            optimizer_d.zero_grad()

            output = netd(normal)
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            fake_img = netg(defect).detach()
            fake_output = netd(fake_img)
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()
            d_loss.update(error_d_real + error_d_fake)

            if opt.debug:
                if not os.path.exists(opt.save_path):
                    os.makedirs(opt.save_path)

                imgs = torch.cat((defect, fake_img), 0)
                tv.utils.save_image(imgs, os.path.join(opt.save_path, '{}_defect_repair.jpg'.format(ii)),
                                    normalize=True,
                                    range=(-1, 1))

        if (ii + 1) % opt.g_every == 0:
            optimizer_g.zero_grad()
            netd.eval()
            fake_img = netg(defect)
            fake_output = netd(fake_img)

            error_g = criterion(fake_output, true_labels)
            error_c = contrast_criterion(normal, fake_img)
            losses = error_g + opt.contrast_loss_weight * error_c
            losses.backward()
            optimizer_g.step()
            g_loss.update(error_g)
            c_loss.update(opt.contrast_loss_weight * error_c)

            progressbar.set_description(
                'Epoch: {}. Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}. Contrast loss: {:.5f}.'.format(
                    epoch, ii, d_loss.getavg(), g_loss.getavg(), c_loss.getavg()))

    scheduler_d.step()
    scheduler_g.step()
