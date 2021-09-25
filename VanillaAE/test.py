from __future__ import print_function
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from networks import MLP
from utils import get_dataloader, print_and_write_log, set_random_seed


parser = argparse.ArgumentParser()
# basic
parser.add_argument('--dataset', required=True, help='folderall | filelist')
parser.add_argument('--dataroot', default='', help='path to dataset')
parser.add_argument('--datalist', default='', help='path to dataset file list')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='dimension of the latent layers')
parser.add_argument('--nblk', type=int, default=2, help='number of blocks')
parser.add_argument('--no_cuda', action='store_true', help='not enable cuda (if use CPU only)')
parser.add_argument('--netG', default='', help="path to netG (optional). It not given, find the checkpoint in '--expf'")
parser.add_argument('--expf', default='./experiments', help='folder to save visualized images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual random seed')

# test
parser.add_argument('--epoch_test', type=int, default=20, help='which epoch to load for test')
parser.add_argument('--eval', action='store_true', help='use eval mode during test time')
parser.add_argument('--resf', type=str, default='./results', help='folder to save test results')
parser.add_argument('--num_test', type=int, default=float('inf'), help='how many images to test')
parser.add_argument('--show_input', action='store_true', help='also save side-by-side results with input (for metric evaluation)')

opt = parser.parse_args()
opt.is_train = False


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Args:
        input_image (torch.tensor): the input tensor array.
        imtype (type): the desired type of the converted numpy image array.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.detach()
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: transpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


assert opt.dataset in ['folderall', 'filelist'], 'Dataset option should be `folderall` or `filelist` in test mode'
assert opt.batchSize == 1, 'Batch size should be 1 in test mode for the moment'

if opt.netG == '':
    opt.netG = os.path.join(opt.expf, 'checkpoints', 'netG_epoch_%03d.pth' % opt.epoch_test)
os.makedirs(os.path.join(opt.expf, 'logs'), exist_ok=True)
test_log_file = os.path.join(opt.expf, 'logs', 'test_log.txt')

cudnn.benchmark = True

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print_and_write_log(test_log_file, "Random Seed: %d" % opt.manualSeed)
set_random_seed(opt.manualSeed)

if torch.cuda.is_available() and opt.no_cuda:
    print_and_write_log(test_log_file, "WARNING: You have a CUDA device, so you should probably run without --no_cuda")

dataloader, nc = get_dataloader(opt)
opt.nc = nc

opt.resf = os.path.join(opt.resf, 'epoch_%03d_seed_%d' % (opt.epoch_test, opt.manualSeed))
os.makedirs(opt.resf, exist_ok=True)
if opt.show_input:
    resfwi = os.path.join(os.path.split(opt.resf)[0], os.path.split(opt.resf)[1] + '_with_input')
    os.makedirs(resfwi, exist_ok=True)
    opt.resfwi = resfwi

print_and_write_log(test_log_file, opt)

device = torch.device("cuda:0" if not opt.no_cuda else "cpu")
nc = int(opt.nc)
imageSize = int(opt.imageSize)
nz = int(opt.nz)
nblk = int(opt.nblk)
model_netG = MLP(input_dim=nc * imageSize * imageSize,
                 output_dim=nc * imageSize * imageSize,
                 dim=nz,
                 n_blk=nblk,
                 norm='none',
                 activ='relu').to(device)
model_netG.load_state_dict(torch.load(opt.netG, map_location=device))
print_and_write_log(test_log_file, 'netG:')
print_and_write_log(test_log_file, str(model_netG))

if opt.eval:
    model_netG.eval()

for i, data in enumerate(tqdm(dataloader), 0):
    img, img_path = data
    img_name = os.path.splitext(os.path.basename(img_path[0]))[0] + '.png'
    if i >= opt.num_test:
        break
    real = img.to(device)
    with torch.no_grad():
        recon = model_netG(real)
    recon_img = tensor2im(recon)
    if opt.show_input:
        real_img = tensor2im(real)
        real_recon_img = np.concatenate([real_img, recon_img], 1)
        real_recon_img_pil = Image.fromarray(real_recon_img)
        real_recon_img_pil.save(os.path.join(opt.resfwi, img_name))
    recon_img_pil = Image.fromarray(recon_img)
    recon_img_pil.save(os.path.join(opt.resf, img_name))

print_and_write_log(test_log_file, 'Finish testing.')
