from __future__ import print_function
import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tqdm import tqdm

from models import VanillaAE
from utils import get_dataloader, print_and_write_log, set_random_seed


parser = argparse.ArgumentParser()
# basic
parser.add_argument('--dataset', required=True, help='folderall | filelist | pairfilelist')
parser.add_argument('--dataroot', default='', help='path to dataset')
parser.add_argument('--datalist', default='', help='path to dataset file list')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='dimension of the latent layers')
parser.add_argument('--nblk', type=int, default=2, help='number of blocks')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--no_cuda', action='store_true', help='not enable cuda (if use CPU only)')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--expf', default='./experiments', help='folder to save visualized images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual random seed')

# display and save
parser.add_argument('--log_iter', type=int, default=50, help='log interval (iterations)')
parser.add_argument('--visualize_iter', type=int, default=500, help='visualization interval (iterations)')
parser.add_argument('--ckpt_save_epoch', type=int, default=1, help='checkpoint save interval (epochs)')

# losses
parser.add_argument('--mse_w', type=float, default=1.0, help='weight for mse (L2) spatial loss')
parser.add_argument('--ffl_w', type=float, default=0.0, help='weight for focal frequency loss')
parser.add_argument('--alpha', type=float, default=1.0, help='the scaling factor alpha of the spectrum weight matrix for flexibility')
parser.add_argument('--patch_factor', type=int, default=1, help='the factor to crop image patches for patch-based focal frequency loss')
parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
parser.add_argument('--log_matrix', action='store_true', help='whether to adjust the spectrum weight matrix by logarithm')
parser.add_argument('--batch_matrix', action='store_true', help='whether to calculate the spectrum weight matrix using batch-based statistics')
parser.add_argument('--freq_start_epoch', type=int, default=1, help='the start epoch to add focal frequency loss')

opt = parser.parse_args()
opt.is_train = True


os.makedirs(os.path.join(opt.expf, 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.expf, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(opt.expf, 'logs'), exist_ok=True)
train_log_file = os.path.join(opt.expf, 'logs', 'train_log.txt')
opt.train_log_file = train_log_file

cudnn.benchmark = True

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print_and_write_log(train_log_file, "Random Seed: %d" % opt.manualSeed)
set_random_seed(opt.manualSeed)

if torch.cuda.is_available() and opt.no_cuda:
    print_and_write_log(train_log_file, "WARNING: You have a CUDA device, so you should probably run without --no_cuda")

dataloader, nc = get_dataloader(opt)
opt.nc = nc

print_and_write_log(train_log_file, opt)

model = VanillaAE(opt)

num_epochs = opt.nepoch
iters = 0

matrix = None
for epoch in tqdm(range(1, num_epochs + 1)):
    for i, data in enumerate(tqdm(dataloader), 0):
        if opt.dataset == 'pairfilelist':
            img, matrix = data
            data = img

        # main training code
        errG_pix, errG_freq = model.gen_update(data, epoch, matrix)

        # logs
        if i % opt.log_iter == 0:
            print_and_write_log(train_log_file,
                                '[%d/%d][%d/%d] LossPixel: %.10f LossFreq: %.10f' %
                                (epoch, num_epochs, i, len(dataloader), errG_pix.item(), errG_freq.item()))

        # write images for visualization
        if (iters % opt.visualize_iter == 0) or ((epoch == num_epochs) and (i == len(dataloader) - 1)):
            real_cpu = data.cpu()
            recon = model.sample(real_cpu)
            visual = torch.cat([real_cpu[:16], recon.detach().cpu()[:16]], 0)
            vutils.save_image(visual, '%s/images/epoch_%03d_real_recon.png' % (opt.expf, epoch), normalize=True, nrow=16)

        iters += 1

    # save checkpoints
    if epoch % opt.ckpt_save_epoch == 0 or epoch == num_epochs:
        model.save_checkpoints('%s/checkpoints' % opt.expf, epoch)

print_and_write_log(train_log_file, 'Finish training.')
