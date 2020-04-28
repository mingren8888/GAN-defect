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
    # data_path = r'/data/sdv2/GAN/data/gan_defect/grid_96/train'
    # val_path = r'/data/sdv2/GAN/data/gan_defect/grid_96/val'
    # save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-2'
    # work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0427-2'
    # val_save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-2-val'

    data_path = r'/data/sdv2/GAN/data/gan_defect/1GE02/train'
    val_path = r'/data/sdv2/GAN/data/gan_defect/1GE02/val'
    save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-1ge02'
    work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0427-1ge02'
    val_save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-1ge02-val'

    with_segmentation = False
    num_workers = 4
    image_size = 128
    batch_size = 16
    max_epoch = 2000
    steps = [1000, 1500, 1800]
    lrg = 1e-4
    lrd = 1e-5
    lrs = 1e-2
    beta1 = 0.5
    use_gpu = True
    nc = 3
    nBottleneck = 4000
    ngf = 64
    ndf = 64
    defect_mode = 'geometry'

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

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    checkpoint_interval = 100

    debug = True
    validate = False


def train(opt):
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        # tv.transforms.ToTensor(),
        DefectAdder(mode=opt.defect_mode, defect_shape=('line',)),
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
        checkpoint = modify_checkpoint(netd, torch.load(opt.netd_path, map_location=map_location)['net'])
        netd.load_state_dict(checkpoint, strict=False)
    if opt.netg_path:
        print('loading checkpoint for generator...')
        checkpoint = modify_checkpoint(netg, torch.load(opt.netg_path, map_location=map_location)['net'])
        netg.load_state_dict(checkpoint, strict=False)

    optimizer_g = optim.Adam(netg.parameters(), opt.lrg, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(netd.parameters(), opt.lrd, betas=(opt.beta1, 0.999))
    optimizer_s = optim.Adam(seg_model.parameters(), opt.lrs, betas=(opt.beta1, 0.999))

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)
    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=opt.steps, gamma=0.1)

    criterion = nn.BCELoss()
    contrast_criterion = nn.MSELoss()

    true_labels = torch.ones(opt.batch_size)
    fake_labels = torch.zeros(opt.batch_size)
    # fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1)
    # noises = torch.randn(opt.batch_size, opt.nz, 1, 1)

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        seg_model.cuda()
        criterion.cuda()
        contrast_criterion.cuda()
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
                error_c = contrast_criterion(fake_img, normal)
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

            progressbar.set_description(
                'Epoch: {}. Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}. Contrast loss: {:.5f}. Segmentation loss: {:.5f}'.format(
                    epoch, ii, d_loss.getavg(), g_loss.getavg(), c_loss.getavg(), s_loss.getavg()))

        scheduler_d.step(epoch=epoch)
        scheduler_g.step(epoch=epoch)
        if epoch >= opt.s_start and opt.with_segmentation:
            scheduler_s.step(epoch=epoch)
        if opt.validate:
            validate(opt, netd, netg, seg_model)

        if (epoch + 1) % opt.checkpoint_interval == 0:
            state_d = {'net': netd.state_dict(), 'optimizer': optimizer_d.state_dict(), 'epoch': epoch}
            state_g = {'net': netg.state_dict(), 'optimizer': optimizer_g.state_dict(), 'epoch': epoch}
            print('saving checkpoints...')
            torch.save(state_d, os.path.join(opt.work_dir, f'd_ckpt_e{epoch + 1}.pth'))
            torch.save(state_g, os.path.join(opt.work_dir, f'g_ckpt_e{epoch + 1}.pth'))


def validate(opt, netd, netg, nets):
    netd.eval()
    netg.eval()
    nets.eval()
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        # tv.transforms.ToTensor(),
        DefectAdder(mode=opt.defect_mode, defect_shape=('line',)),
        ToTensorList(),
        NormalizeList(opt.mean, opt.std),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.val_path, transform=transforms)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            drop_last=True)

    progressbar = tqdm(dataloader)
    for ii, (imgs, _) in enumerate(progressbar):
        normal, defect, target = imgs
        if opt.use_gpu:
            normal = normal.cuda()
            defect = defect.cuda()
            target = target.cuda()
        repair = netg(defect)
        if opt.with_segmentation:
            seg_input = torch.cat([defect, repair], dim=1)
            seg = nets(seg_input)
        else:
            seg = None

        if opt.with_segmentation:
            metrics = []
            lbl_pred = seg.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                label_accuracy_score(
                    lbl_true, lbl_pred, n_class=2)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)
            progressbar.set_description(
                f'Acc: {metrics[0]:.5f}, Acc_cls: {metrics[1]:.5f}, MIU: {metrics[2]:.5f}, Fwavacc: {metrics[3]:.5f}')
        if opt.debug:
            if not os.path.exists(opt.val_save_path):
                os.makedirs(opt.val_save_path)

            imgs = torch.cat((defect, repair), 0)
            tv.utils.save_image(imgs, os.path.join(opt.val_save_path, '{}_defect_repair.jpg'.format(ii)),
                                normalize=True,
                                range=(-1, 1))


if __name__ == '__main__':
    opt = Config()
    train(opt)
