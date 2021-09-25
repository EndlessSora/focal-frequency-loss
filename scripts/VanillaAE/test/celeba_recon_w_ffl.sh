#!/usr/bin/env bash

set -x

VERSION='celeba_recon_w_ffl'
DATA='filelist'
DATAROOT='./datasets/celeba'
DATALIST='./datasets/celeba_recon_lists/test.txt'
EXP_DIR='./VanillaAE/experiments/'$VERSION
RES_DIR='./VanillaAE/results/'$VERSION
WORKER=0

python ./VanillaAE/test.py \
    --dataset $DATA \
    --dataroot $DATAROOT \
    --datalist $DATALIST \
    --workers $WORKER \
    --batchSize 1 \
    --imageSize 64 \
    --nz 256 \
    --nblk 2 \
    --expf $EXP_DIR \
    --manualSeed 1112 \
    --epoch_test 20 \
    --eval \
    --resf $RES_DIR \
    --show_input
