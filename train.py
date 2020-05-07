import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm.autonotebook import tqdm
from model import *
from fcn import *
from defect import DefectAdder, NormalizeList, ToTensorList
from utils import *
from loss import *
import os
from trainer import *


class Config(object):
    data_path = r'/data/sdv2/GAN/data/gan_defect/grid_96/train'
    val_path = r'/data/sdv2/GAN/data/gan_defect/grid_96/val'
    save_path = '/data/sdv2/GAN/GAN_defect/imgs/0430'
    work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0430'
    val_save_path = '/data/sdv2/GAN/GAN_defect/imgs/0430-val'

    # data_path = r'/data/sdv2/GAN/data/gan_defect/1GE02/train'
    # val_path = r'/data/sdv2/GAN/data/gan_defect/1GE02/val'
    # save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-1ge02'
    # work_dir = '/data/sdv2/GAN/GAN_defect/workdirs/0427-1ge02'
    # val_save_path = '/data/sdv2/GAN/GAN_defect/imgs/0427-1ge02-val'

    num_workers = 4
    image_size = 128
    batch_size = 16
    max_epoch = 300
    steps = [100, 200]
    lrg = 1e-3
    lrd = 1e-4
    lrs = 1e-2
    beta1 = 0.5

    nBottleneck = 4000
    nc = 3
    ngf = 64
    ndf = 64
    defect_mode = 'geometry'

    contrast_loss_weight = 1

    # device settings
    use_gpu = True
    gpus = 1
    nodes = 1
    nr = 0

    d_every = 1
    g_every = 1
    s_every = 5
    s_start = 0
    decay_every = 10
    netd_path = '/data/sdv2/GAN/GAN_defect/workdirs/0427-1ge02/d_ckpt_e2000.pth'
    netg_path = '/data/sdv2/GAN/GAN_defect/workdirs/0427-1ge02/g_ckpt_e2000.pth'
    # netd_path = None
    # netg_path = None

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    checkpoint_interval = 100

    debug = True
    validate = True
    with_segmentation = False


def main(opt):
    if opt.use_gpu and opt.gpus > 1:
        print('distributed training')
        os.environ['MASTER_ADDR'] = '172.27.9.82'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(distributed_train, nprocs=opt.gpus, args=(opt,))
    else:
        print('undistributed training')
        train(opt)


def train(opt):
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        # tv.transforms.ToTensor(),
        DefectAdder(mode=opt.defect_mode, defect_shape=('line',), normal_only=True),
        ToTensorList(),
        NormalizeList(opt.mean, opt.std),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    train_dataloader = DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  drop_last=True)

    if opt.validate:
        val_transforms = tv.transforms.Compose([
            tv.transforms.Resize(opt.image_size),
            tv.transforms.CenterCrop(opt.image_size),
            # tv.transforms.ToTensor(),
            DefectAdder(mode=opt.defect_mode, defect_shape=('line',)),
            ToTensorList(),
            NormalizeList(opt.mean, opt.std),
            # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        val_dataset = tv.datasets.ImageFolder(opt.val_path, transform=val_transforms)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=True,
                                    num_workers=opt.num_workers,
                                    drop_last=True)
    else:
        val_dataloader = None

    map_location = lambda storage, loc: storage
    netd = Discriminator(opt)
    netg = Generater(opt)
    nets = FCN32s(n_class=2, input_channels=6)

    if opt.use_gpu:
        netd.cuda()
        netg.cuda()
        nets.cuda()

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
    optimizer_s = optim.Adam(nets.parameters(), opt.lrs, betas=(opt.beta1, 0.999))

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)
    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=opt.steps, gamma=0.1)

    trainer = Trainer(opt, [netd, netg, nets], [optimizer_d, optimizer_g, optimizer_s],
                      [scheduler_d, scheduler_g, scheduler_s],
                      train_dataloader, val_dataloader)
    trainer.train()


def distributed_train(gpu, opt):
    rank = opt.nr * opt.gpus + gpu
    world_size = opt.gpus * opt.nodes
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    torch.cuda.set_device(gpu)

    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        # tv.transforms.ToTensor(),
        DefectAdder(mode=opt.defect_mode, defect_shape=('line',), normal_only=True),
        ToTensorList(),
        NormalizeList(opt.mean, opt.std),
        # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_dataloader = DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    if opt.validate:
        val_transforms = tv.transforms.Compose([
            tv.transforms.Resize(opt.image_size),
            tv.transforms.CenterCrop(opt.image_size),
            # tv.transforms.ToTensor(),
            DefectAdder(mode=opt.defect_mode, defect_shape=('line',)),
            ToTensorList(),
            NormalizeList(opt.mean, opt.std),
            # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        val_dataset = tv.datasets.ImageFolder(opt.val_path, transform=val_transforms)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                      num_replicas=world_size,
                                                                      rank=rank)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    num_workers=opt.num_workers,
                                    drop_last=True,
                                    sampler=val_sampler
                                    )
    else:
        val_dataloader = None

    map_location = lambda storage, loc: storage
    netd = Discriminator(opt)
    netg = Generater(opt)
    nets = FCN32s(n_class=2, input_channels=6)

    if opt.use_gpu:
        netd.cuda(gpu)
        netg.cuda(gpu)
        nets.cuda(gpu)

    netd = nn.parallel.DistributedDataParallel(netd, device_ids=[gpu])
    netg = nn.parallel.DistributedDataParallel(netg, device_ids=[gpu])
    nets = nn.parallel.DistributedDataParallel(nets, device_ids=[gpu])

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
    optimizer_s = optim.Adam(nets.parameters(), opt.lrs, betas=(opt.beta1, 0.999))

    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=opt.steps, gamma=0.1)
    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=opt.steps, gamma=0.1)
    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=opt.steps, gamma=0.1)

    criterion = nn.BCELoss()
    contrast_criterion = nn.MSELoss()

    true_labels = torch.ones(opt.batch_size)
    fake_labels = torch.zeros(opt.batch_size)

    if opt.use_gpu:
        criterion.cuda()
        contrast_criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        # fix_noises, noises = fix_noises.cuda(), noises.cuda()

    trainer = Trainer(opt, [netd, netg, nets], [optimizer_d, optimizer_g, optimizer_s],
                      [scheduler_d, scheduler_g, scheduler_s],
                      train_dataloader, val_dataloader)
    trainer.train()


if __name__ == '__main__':
    opt = Config()
    main(opt)
