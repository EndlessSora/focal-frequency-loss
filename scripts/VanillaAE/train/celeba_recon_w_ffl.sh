#!/usr/bin/env bash

set -x

VERSION='celeba_recon_w_ffl'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/train.txt'
EXP_DIR='./VanillaAE/experiments/'$VERSION
WORKER=8

python ./VanillaAE/train.py \
    --dataset $DATA \
    --dataroot $DATAROOT \
    --datalist $DATALIST \
    --workers $WORKER \
    --batchSize 128 \
    --imageSize 64 \
    --nz 256 \
    --nblk 2 \
    --nepoch 20 \
    --expf $EXP_DIR \
    --manualSeed 1112 \
    --log_iter 50 \
    --visualize_iter 500 \
    --ckpt_save_epoch 1 \
    --mse_w 1.0 \
    --ffl_w 100.0
