#!/usr/bin/env bash

set -x

VERSION='celeba_recon_wo_ffl'
EPOCH='epoch_020_seed_1112_with_input'
METRICS='psnr ssim lpips fid lfd'
FDRF='./VanillaAE/results/'$VERSION'/'$EPOCH
LOGS='./VanillaAE/results/'$VERSION'/metrics_'$EPOCH'.txt'

python ./metrics/calc_metrics.py \
    --metrics $METRICS \
    --fdrf $FDRF \
    --logs $LOGS
