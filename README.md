## Focal Frequency Loss - Official PyTorch Implementation

![teaser](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/teaser.jpg)

This repository provides the official PyTorch implementation for the following paper:

**Focal Frequency Loss for Image Reconstruction and Synthesis**<br>
[Liming Jiang](https://liming-jiang.com/), [Bo Dai](http://daibo.info/), [Wayne Wu](https://wywu.github.io/) and [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)<br>
In ICCV 2021.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/ffl/index.html) | [**Paper**](https://arxiv.org/abs/2012.12821)
> **Abstract:** *Image reconstruction and synthesis have witnessed remarkable progress thanks to the development of generative models. Nonetheless, gaps could still exist between the real and generated images, especially in the frequency domain. In this study, we show that narrowing gaps in the frequency domain can ameliorate image reconstruction and synthesis quality further. We propose a novel focal frequency loss, which allows a model to adaptively focus on frequency components that are hard to synthesize by down-weighting the easy ones. This objective function is complementary to existing spatial losses, offering great impedance against the loss of important frequency information due to the inherent bias of neural networks. We demonstrate the versatility and effectiveness of focal frequency loss to improve popular models, such as VAE, pix2pix, and SPADE, in both perceptual quality and quantitative performance. We further show its potential on StyleGAN2.*

## Updates

- [09/2021] The **code** of Focal Frequency Loss is **released**.

- [07/2021] The [paper](https://arxiv.org/abs/2012.12821) of Focal Frequency Loss is accepted by **ICCV 2021**.

## Quick Start

Run `pip install focal-frequency-loss` for installation. Then, the following code is all you need.

```python
from focal_frequency_loss import FocalFrequencyLoss as FFL
ffl = FFL(loss_weight=1.0, alpha=1.0)  # initialize nn.Module class

import torch
fake = torch.randn(4, 3, 64, 64)  # replace it with the predicted tensor of shape (N, C, H, W)
real = torch.randn(4, 3, 64, 64)  # replace it with the target tensor of shape (N, C, H, W)

loss = ffl(fake, real)  # calculate focal frequency loss
```

**Tips:** 

1. Current supported PyTorch version: `torch<=1.7.1,>=1.1.0`. Warnings can be ignored.
2. Arguments to initialize the `FocalFrequencyLoss` class:
	- `loss_weight (float)`: weight for focal frequency loss. Default: 1.0
	- `alpha (float)`: the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
	- `patch_factor (int)`: the factor to crop image patches for patch-based focal frequency loss. Default: 1
	- `ave_spectrum (bool)`: whether to use minibatch average spectrum. Default: False
	- `log_matrix (bool)`: whether to adjust the spectrum weight matrix by logarithm. Default: False
	- `batch_matrix (bool)`: whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
3. Experience shows that the main hyperparameters you need to adjust are `loss_weight` and `alpha`. The loss weight may always need to be adjusted first. Then, a larger alpha indicates that the model is more focused. We use `alpha=1.0` as default.

## Exmaple: Image Reconstruction (Vanilla AE)

As a guide, we provide an example of applying the proposed focal frequency loss (FFL) for Vanilla AE image reconstruction on CelebA. Applying FFL is pretty easy. The core details can be found [here](https://github.com/EndlessSora/focal-frequency-loss/blob/master/VanillaAE/models.py).

### Installation

After installing [Anaconda](https://www.anaconda.com/), we recommend you to create a new conda environment with python 3.8.3:

```bash
conda create -n ffl python=3.8.3 -y
conda activate ffl
```

Clone this repo, install PyTorch 1.4.0 (`torch<=1.7.1,>=1.1.0` may also work) and other dependencies:

```bash
git clone https://github.com/EndlessSora/focal-frequency-loss.git
cd focal-frequency-loss
pip install -r VanillaAE/requirements.txt
```

### Dataset Preparation

In this example, please download [img\_align\_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) of the CelebA dataset from its [official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Then, we highly recommend you to unzip this file and symlink the `img_align_celeba` folder to `./datasets/celeba` by:

```bash
bash scripts/datasets/prepare_celeba.sh [PATH_TO_IMG_ALIGN_CELEBA]
```

Or you can simply move the `img_align_celeba` folder to `./datasets/celeba`. The resulting directory structure should be:

```
├── datasets
│    ├── celeba
│    │    ├── img_align_celeba  
│    │    │    ├── 000001.jpg
│    │    │    ├── 000002.jpg
│    │    │    ├── 000003.jpg
│    │    │    ├── ...
```

### Test and Evaluation Metrics

Download the [pretrained models](https://drive.google.com/file/d/1YIH09eoDyP2JLmiYJpju4hOkVFO7M3b_/view?usp=sharing) and unzip them to `./VanillaAE/experiments`.

We have provided the example [test scripts](https://github.com/EndlessSora/focal-frequency-loss/tree/master/scripts/VanillaAE/test). If you only have a CPU environment, please specify `--no_cuda` in the script. Run:

```bash
bash scripts/VanillaAE/test/celeba_recon_wo_ffl.sh
bash scripts/VanillaAE/test/celeba_recon_w_ffl.sh
```

The Vanilla AE image reconstruction results will be saved at `./VanillaAE/results` by default.

After testing, you can further calculate the evaluation metrics for this example. We have implemented a series of [evaluation metrics](https://github.com/EndlessSora/focal-frequency-loss/tree/master/metrics) we used and provided the [metric scripts](https://github.com/EndlessSora/focal-frequency-loss/tree/master/scripts/VanillaAE/metrics). Run:

```bash
bash scripts/VanillaAE/metrics/celeba_recon_wo_ffl.sh
bash scripts/VanillaAE/metrics/celeba_recon_w_ffl.sh
```

You will see the scores of different metrics. The metric logs will be saved in the respective experiment folders at `./VanillaAE/results`.

### Training

We have provided the example [training scripts](https://github.com/EndlessSora/focal-frequency-loss/tree/master/scripts/VanillaAE/train). If you only have a CPU environment, please specify `--no_cuda` in the script. Run:

```bash
bash scripts/VanillaAE/train/celeba_recon_wo_ffl.sh
bash scripts/VanillaAE/train/celeba_recon_w_ffl.sh 
```

After training, inference on the newly trained models is similar to [Test and Evaluation Metrics](#test-and-evaluation-metrics). The results could be better reproduced on NVIDIA Tesla V100 GPUs.

## More Results

Here, we show other examples of applying the proposed focal frequency loss (FFL) under diverse settings.

### Image Reconstruction (VAE)

![reconvae](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/reconvae.jpg)

### Image-to-Image Translation (pix2pix | SPADE)

![consynI2I](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/consynI2I.jpg)

### Unconditional Image Synthesis (StyleGAN2)

256x256 results (without truncation) and the mini-batch average spectra (adjusted to better contrast):

![unsynsg2res256](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/unsynsg2res256.jpg)

1024x1024 results (without truncation) synthesized by StyleGAN2 with FFL:

![unsynsg2res1024](https://raw.githubusercontent.com/EndlessSora/focal-frequency-loss/master/resources/unsynsg2res1024.jpg)

## Citation

If you find this work useful for your research, please cite our paper:

```
@inproceedings{jiang2021focal,
  title={Focal Frequency Loss for Image Reconstruction and Synthesis},
  author={Jiang, Liming and Dai, Bo and Wu, Wayne and Loy, Chen Change},
  booktitle={ICCV},
  year={2021}
}
```

## Acknowledgments

The code of Vanilla AE is inspired by [PyTorch DCGAN](https://github.com/pytorch/examples/tree/master/dcgan) and [MUNIT](https://github.com/NVlabs/MUNIT). Part of the evaluation metric code is borrowed from [MMEditing](https://github.com/open-mmlab/mmediting). We also apply [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid) as evaluation metrics.

## License

All rights reserved. The code is released under the [MIT License](https://github.com/EndlessSora/focal-frequency-loss/blob/master/LICENSE.md).

Copyright (c) 2021
