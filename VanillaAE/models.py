import os

import torch
import torch.nn as nn
import torch.optim as optim
from focal_frequency_loss import FocalFrequencyLoss as FFL

from networks import MLP
from utils import print_and_write_log, weights_init


class VanillaAE(nn.Module):
    def __init__(self, opt):
        super(VanillaAE, self).__init__()
        self.opt = opt
        self.device = torch.device("cuda:0" if not opt.no_cuda else "cpu")
        nc = int(opt.nc)
        imageSize = int(opt.imageSize)
        nz = int(opt.nz)
        nblk = int(opt.nblk)

        # generator
        self.netG = MLP(input_dim=nc * imageSize * imageSize,
                        output_dim=nc * imageSize * imageSize,
                        dim=nz,
                        n_blk=nblk,
                        norm='none',
                        activ='relu').to(self.device)
        weights_init(self.netG)
        if opt.netG != '':
            self.netG.load_state_dict(torch.load(opt.netG, map_location=self.device))
        print_and_write_log(opt.train_log_file, 'netG:')
        print_and_write_log(opt.train_log_file, str(self.netG))

        # losses
        self.criterion = nn.MSELoss()
        # define focal frequency loss
        self.criterion_freq = FFL(loss_weight=opt.ffl_w,
                                  alpha=opt.alpha,
                                  patch_factor=opt.patch_factor,
                                  ave_spectrum=opt.ave_spectrum,
                                  log_matrix=opt.log_matrix,
                                  batch_matrix=opt.batch_matrix).to(self.device)

        # misc
        self.to(self.device)

        # optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def forward(self):
        pass

    def gen_update(self, data, epoch, matrix=None):
        self.netG.zero_grad()
        real = data.to(self.device)
        if matrix is not None:
            matrix = matrix.to(self.device)
        recon = self.netG(real)

        # apply pixel-level loss
        errG_pix = self.criterion(recon, real) * self.opt.mse_w

        # apply focal frequency loss
        if epoch >= self.opt.freq_start_epoch:
            errG_freq = self.criterion_freq(recon, real, matrix)
        else:
            errG_freq = torch.tensor(0.0).to(self.device)

        errG = errG_pix + errG_freq
        errG.backward()
        self.optimizerG.step()

        return errG_pix, errG_freq

    def sample(self, x):
        x = x.to(self.device)
        self.netG.eval()
        with torch.no_grad():
            recon = self.netG(x)
        self.netG.train()

        return recon

    def save_checkpoints(self, ckpt_dir, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (ckpt_dir, epoch))
