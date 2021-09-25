#!/usr/bin/env bash

set -x

PATH_TO_IMG_ALIGN_CELEBA=$1

mkdir -p ./datasets/celeba
ln -s $PATH_TO_IMG_ALIGN_CELEBA ./datasets/celeba
