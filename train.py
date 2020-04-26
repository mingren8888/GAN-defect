import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
from tqdm.autonotebook import tqdm
from model import *
from fcn import *
from defect import DefectAdder, NormalizeList, ToTensorList
from utils import *
from loss import *
import os


class Config(object):
    data_path = r'/data/sdv2/GAN/data/gan_defect/GRID'
    save_path = '/data/sdv2/GAN/GAN_defect/imgs/0426-1'
    work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0426-1'

    with_segmentation = True
    num_workers = 4
    image_size = 96
    batch_size = 16
    max_epoch = 150
    steps = [100, 130]
    lrg = 1e-5
    lrd = 1e-5
    lrs = 1e-2
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
    s_every = 5
    s_start = 0
    decay_every = 10
    netd_path = '/data/sdv2/GAN/GAN_defect/workdirs/0424_grid_1/d_ckpt_e1000.pth'
    netg_path = '/data/sdv2/GAN/GAN_defect/workdirs/0424_grid_1/g_ckpt_e1000.pth'
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


def train(opt):
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
    seg_model = FCN32s(n_class=2, input_channels=6)

    if opt.netd_path:
        print('loading checkpoint for discriminator...')
        netd.load_state_dict(torch.load(opt.netd_path, map_location=map_location)['net'])
    if opt.netg_path:
        print('loading checkpoint for generator...')
        netg.load_state_dict(torch.load(opt.netg_path, map_location=map_location)['net'])

    optimizer_g = optim.Adam(netg.parameters(), opt.lrg, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(netd.parameters(), opt.lrd, betas=(opt.beta1, 0.999))
    optimizer_s = optim.Adam(seg_model.parameters(), opt.lrs, betas=(opt.beta1, 0.999))

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)
    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=opt.steps, gamma=0.1)

    criterion = nn.BCELoss()
    contrast_criterion = nn.L1Loss(reduction='mean')

    true_labels = torch.ones(opt.batch_size)
    fake_labels = torch.zeros(opt.batch_size)
    # fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1)
    # noises = torch.randn(opt.batch_size, opt.nz, 1, 1)

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        seg_model.cuda()
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
        s_loss = AverageMeter()
        if (epoch + 1) % opt.checkpoint_interval == 0:
            state_d = {'net': netd.state_dict(), 'optimizer': optimizer_d.state_dict(), 'epoch': epoch}
            state_g = {'net': netg.state_dict(), 'optimizer': optimizer_g.state_dict(), 'epoch': epoch}
            print('saving checkpoints...')
            torch.save(state_d, os.path.join(opt.work_dir, f'd_ckpt_e{epoch + 1}.pth'))
            torch.save(state_g, os.path.join(opt.work_dir, f'g_ckpt_e{epoch + 1}.pth'))
        for ii, (imgs, _) in enumerate(progressbar):
            normal, defect, target = imgs
            if opt.use_gpu:
                normal = normal.cuda()
                defect = defect.cuda()
                target = target.cuda()

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


            if (ii + 1) % opt.g_every == 0:
                # train generator
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

                if opt.debug:
                    if not os.path.exists(opt.save_path):
                        os.makedirs(opt.save_path)

                    imgs = torch.cat((defect, fake_img), 0)
                    tv.utils.save_image(imgs, os.path.join(opt.save_path, '{}_defect_repair.jpg'.format(ii)),
                                        normalize=True,
                                        range=(-1, 1))

            if epoch >= opt.s_start and (ii + 1) % opt.s_every == 0 and opt.with_segmentation:
                optimizer_s.zero_grad()
                fake_img = netg(defect).detach()
                seg_input = torch.cat([defect, fake_img], dim=1)
                seg_output = seg_model(seg_input)
                target = target.long()
                loss = cross_entropy2d(seg_output, target)
                loss /= len(defect)
                loss.backward()
                optimizer_s.step()
                s_loss.update(loss)

                metrics = []
                lbl_pred = seg_output.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu().numpy()
                acc, acc_cls, mean_iu, fwavacc = \
                    label_accuracy_score(
                        lbl_true, lbl_pred, n_class=2)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
                metrics = np.mean(metrics, axis=0)
                # print(metrics)


                # if opt.debug:
                #     if not os.path.exists(opt.save_path):
                #         os.makedirs(opt.save_path)
                #     target = target.float()
                #     segs = torch.cat((target, seg_output), 0)
                #     tv.utils.save_image(segs, os.path.join(opt.save_path, '{}_seg_result.jpg'.format(ii)),
                #                         normalize=True,
                #                         range=(-1, 1))

            progressbar.set_description(
                'Epoch: {}. Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}. Contrast loss: {:.5f}. Segmentation loss: {:.5f}'.format(
                    epoch, ii, d_loss.getavg(), g_loss.getavg(), c_loss.getavg(), s_loss.getavg()))

        scheduler_d.step(epoch=epoch)
        scheduler_g.step(epoch=epoch)
        if epoch >= opt.s_start and opt.with_segmentation:
            scheduler_s.step(epoch=epoch)





if __name__ == '__main__':
    opt = Config()
    train(opt)
