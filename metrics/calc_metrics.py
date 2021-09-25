import argparse
import os
import shutil
import time

import cv2
import numpy as np
from tqdm import tqdm

from metric_utils import psnr, ssim, lpips, fid, lfd, mse, print_and_write_log


parser = argparse.ArgumentParser()
# basic
parser.add_argument('--metrics', type=str, default='', nargs='+', help='space separated metrics list: psnr | ssim | lpips | fid | lfd | mse')
parser.add_argument('--logs', type=str, default=None, help='path to log file to record scores')

# 1. single sample test
parser.add_argument('--imgf', type=str, default=None, help='path to fake image')
parser.add_argument('--imgr', type=str, default=None, help='path to real image')

# 2. multiple sample test (reals and fakes in separate folders, corresponding pairs in the same name)
parser.add_argument('--fdf', type=str, default=None, help='path to folder of fake images')
parser.add_argument('--fdr', type=str, default=None, help='path to folder of real images')

# 3. multiple sample test (reals and fakes in one folder, concatenated in the width dimension)
parser.add_argument('--fdrf', type=str, default=None, help='path to folder of real and fake image pairs')

opt = parser.parse_args()


for i in range(len(opt.metrics)):
    opt.metrics[i] = opt.metrics[i].lower()
    assert opt.metrics[i] in ['psnr', 'ssim', 'lpips', 'fid', 'lfd', 'mse'], (
        f'Current supported metrics are: psnr | ssim | lpips | fid | lfd | mse, but got {opt.metrics[i]}')

if opt.logs is not None:
    root = os.path.split(opt.logs)[0]
    root = '.' if root == '' else '.'
    os.makedirs(root, exist_ok=True)

# 1. single sample test
if opt.imgf is not None and opt.imgr is not None:
    assert opt.fdf is None and opt.fdr is None and opt.fdrf is None
    assert 'fid' not in opt.metrics, 'FID should be tested on two sets of images'
    img_fake = cv2.imread(opt.imgf).astype(np.float64)
    img_real = cv2.imread(opt.imgr).astype(np.float64)
    for metric in opt.metrics:
        result = eval(metric + '(img_fake, img_real)')
        print_and_write_log('%s: %.10f' % (metric.upper(), result), opt.logs)

# 2. multiple sample test (reals and fakes in separate folders, corresponding pairs in the same name)
if opt.fdf is not None and opt.fdr is not None:
    assert opt.imgf is None and opt.imgr is None and opt.fdrf is None
    # a dict of list to save metric results
    results = dict()
    for metric in opt.metrics:
        results[metric] = list()
    metrics = opt.metrics.copy()
    if 'fid' in metrics:
        results['fid'].append(fid([opt.fdf, opt.fdr]))
        metrics.remove('fid')
    if len(metrics) > 0:
        listf = [os.path.join(opt.fdf, p) for p in sorted(os.listdir(opt.fdf))]
        listr = [os.path.join(opt.fdr, p) for p in sorted(os.listdir(opt.fdr))]
        assert len(listf) == len(listr)
        for i in tqdm(range(len(listf))):
            assert os.path.basename(listf[i]) == os.path.basename(listr[i])
            img_fake = cv2.imread(listf[i]).astype(np.float64)
            img_real = cv2.imread(listr[i]).astype(np.float64)
            for metric in metrics:
                result = eval(metric + '(img_fake, img_real)')
                results[metric].append(result)
    for metric in opt.metrics:
        result_mean = np.mean(results[metric])
        print_and_write_log('%s: %.10f' % (metric.upper(), result_mean), opt.logs)

# 3. multiple sample test (reals and fakes in one folder, concatenated in the width dimension)
if opt.fdrf is not None:
    assert opt.imgf is None and opt.imgr is None and opt.fdf is None and opt.fdr is None
    # a dict of list to save metric results
    results = dict()
    for metric in opt.metrics:
        results[metric] = list()
    if 'fid' in opt.metrics:
        tmp_path = time.strftime('tmp_%Y%m%d_%H%M%S', time.localtime())
        os.makedirs(f'{tmp_path}/fake', exist_ok=True)
        os.makedirs(f'{tmp_path}/real', exist_ok=True)
    listrf = [os.path.join(opt.fdrf, p) for p in sorted(os.listdir(opt.fdrf))]
    for i in tqdm(range(len(listrf))):
        img = cv2.imread(listrf[i]).astype(np.float64)
        _, w, _ = img.shape
        img_fake = img[:, w // 2:, :]
        img_real = img[:, :w // 2, :]
        for metric in opt.metrics:
            if metric == 'fid':
                cv2.imwrite(f'{tmp_path}/fake/{i:08d}.png', img_fake.astype(np.uint8))
                cv2.imwrite(f'{tmp_path}/real/{i:08d}.png', img_real.astype(np.uint8))
            else:
                result = eval(metric + '(img_fake, img_real)')
                results[metric].append(result)
    if 'fid' in opt.metrics:
        results['fid'].append(fid([f'{tmp_path}/fake', f'{tmp_path}/real']))
        shutil.rmtree(tmp_path)
    for metric in opt.metrics:
        result_mean = np.mean(results[metric])
        print_and_write_log('%s: %.10f' % (metric.upper(), result_mean), opt.logs)
