import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import os
from utils import *
from tqdm.autonotebook import tqdm
import torchvision as tv


class Trainer():
    def __init__(self, opt, model, optimizer, lr_schedule, train_data_loader,
                 valid_data_loader=None, start_epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.opt = opt

        self.cur_epoch = start_epoch

        self.netd, self.netg, self.nets = model
        self.optimizer_d, self.optimizer_g, self.optimizer_s = optimizer
        self.scheduler_d, self.scheduler_g, self.scheduler_s = lr_schedule

        self.criterion = nn.BCELoss()
        self.contrast_criterion = nn.MSELoss()

    def train(self):
        if not os.path.exists(self.opt.work_dir):
            os.makedirs(self.opt.work_dir)

        true_labels = torch.ones(self.opt.batch_size)
        fake_labels = torch.zeros(self.opt.batch_size)

        if self.opt.use_gpu:
            self.criterion.cuda()
            self.contrast_criterion.cuda()
            true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()

        for epoch in range(self.opt.max_epoch):
            progressbar = tqdm(self.train_data_loader)
            d_loss = AverageMeter()
            g_loss = AverageMeter()
            c_loss = AverageMeter()
            s_loss = AverageMeter()
            for ii, (imgs, _) in enumerate(progressbar):
                normal, defect, target = imgs
                if self.opt.use_gpu:
                    normal = normal.cuda()
                    defect = defect.cuda()
                    target = target.cuda()

                if (ii + 1) % self.opt.d_every == 0:
                    # train discriminator
                    self.netd.train()
                    self.optimizer_d.zero_grad()

                    output = self.netd(normal)
                    error_d_real = self.criterion(output, true_labels)
                    error_d_real.backward()

                    fake_img = self.netg(defect).detach()
                    fake_output = self.netd(fake_img)
                    error_d_fake = self.criterion(fake_output, fake_labels)
                    error_d_fake.backward()
                    self.optimizer_d.step()
                    d_loss.update(error_d_real + error_d_fake)

                if (ii + 1) % self.opt.g_every == 0:
                    # train generator
                    self.optimizer_g.zero_grad()
                    self.netd.eval()
                    fake_img = self.netg(defect)
                    fake_output = self.netd(fake_img)

                    error_g = self.criterion(fake_output, true_labels)
                    error_c = self.contrast_criterion(fake_img, normal)
                    losses = error_g + self.opt.contrast_loss_weight * error_c
                    losses.backward()
                    self.optimizer_g.step()
                    g_loss.update(error_g)
                    c_loss.update(self.opt.contrast_loss_weight * error_c)

                    if self.opt.debug:
                        if not os.path.exists(self.opt.save_path):
                            os.makedirs(self.opt.save_path)

                        imgs = torch.cat((defect, fake_img), 0)
                        tv.utils.save_image(imgs, os.path.join(self.opt.save_path, '{}_defect_repair.jpg'.format(ii)),
                                            normalize=True,
                                            range=(-1, 1))

                # if epoch >= opt.s_start and (ii + 1) % opt.s_every == 0 and opt.with_segmentation:
                #     optimizer_s.zero_grad()
                #     fake_img = netg(defect).detach()
                #     seg_input = torch.cat([defect, fake_img], dim=1)
                #     seg_output = seg_model(seg_input)
                #     target = target.long()
                #     loss = cross_entropy2d(seg_output, target)
                #     loss /= len(defect)
                #     loss.backward()
                #     optimizer_s.step()
                #     s_loss.update(loss)

                progressbar.set_description(
                    'Epoch: {}. Step: {}. Discriminator loss: {:.5f}. Generator loss: {:.5f}. Contrast loss: {:.5f}. Segmentation loss: {:.5f}'.format(
                        epoch, ii, d_loss.getavg(), g_loss.getavg(), c_loss.getavg(), s_loss.getavg()))

            self.scheduler_d.step(epoch=epoch)
            self.scheduler_g.step(epoch=epoch)
            if epoch >= self.opt.s_start and self.opt.with_segmentation:
                self.scheduler_s.step(epoch=epoch)
            if self.opt.validate:
                self.validate()

            if (epoch + 1) % self.opt.checkpoint_interval == 0:
                state_d = {'net': self.netd.state_dict(), 'optimizer': self.optimizer_d.state_dict(), 'epoch': epoch}
                state_g = {'net': self.netg.state_dict(), 'optimizer': self.optimizer_g.state_dict(), 'epoch': epoch}
                print('saving checkpoints...')
                torch.save(state_d, os.path.join(self.opt.work_dir, f'd_ckpt_e{epoch + 1}.pth'))
                torch.save(state_g, os.path.join(self.opt.work_dir, f'g_ckpt_e{epoch + 1}.pth'))

    def validate(self):
        self.netd.eval()
        self.netg.eval()
        self.nets.eval()

        progressbar = tqdm(self.valid_data_loader)
        for ii, (imgs, _) in enumerate(progressbar):
            normal, defect, target = imgs
            if self.opt.use_gpu:
                normal = normal.cuda()
                defect = defect.cuda()
                target = target.cuda()
            repair = self.netg(defect)
            if self.opt.with_segmentation:
                seg_input = torch.cat([defect, repair], dim=1)
                seg = self.nets(seg_input)
            else:
                seg = None

            if self.opt.with_segmentation:
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
            if self.opt.debug:
                if not os.path.exists(self.opt.val_save_path):
                    os.makedirs(self.opt.val_save_path)

                imgs = torch.cat((defect, repair), 0)
                tv.utils.save_image(imgs, os.path.join(self.opt.val_save_path, '{}_defect_repair.jpg'.format(ii)),
                                    normalize=True,
                                    range=(-1, 1))
