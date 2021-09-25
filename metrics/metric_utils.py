#######################################################################################
# Part of code from
# https://github.com/open-mmlab/mmediting/blob/master/mmedit/core/evaluation/metrics.py
#######################################################################################

import cv2
import lpips as lpips_ori
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_fid import fid_score


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


# predefine transforms and network for LPIPS calculation
transform = None
loss_fn_alex = None
device = None

def lpips(img1, img2, crop_border=0, input_order='HWC', channel_order='BGR'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Ref:
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
    In CVPR 2018. <https://arxiv.org/pdf/1801.03924.pdf>

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LPIPS calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        channel_order (str): Whether the channel order is 'BGR' or 'RGB'.
            Default: 'BGR'.

    Returns:
        float: lpips result.
    """

    def init():
        global transform, loss_fn_alex, device
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        loss_fn_alex = lpips_ori.LPIPS(net='alex')
        if torch.cuda.is_available():
            loss_fn_alex.cuda()
        device = next(loss_fn_alex.parameters()).device

    # initialize the predefined transforms and network at the first call
    if transform is None or loss_fn_alex is None or device is None:
        init()

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    if channel_order not in ['BGR', 'RGB']:
        raise ValueError(
            f'Wrong channel_order {channel_order}. Supported channel_orders are '
            '"BGR" and "RGB"')
    # input image should be adjusted to RGB for LPIPS calculation
    if channel_order == 'BGR':
        img1, img2 = img1[:, :, ::-1].copy(), img2[:, :, ::-1].copy()

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    tensor1 = transform(img1.astype(np.uint8)).unsqueeze(0).to(device)
    tensor2 = transform(img2.astype(np.uint8)).unsqueeze(0).to(device)
    return loss_fn_alex(tensor1, tensor2).item()


def fid(paths, batch_size=50, device=None, dims=2048):
    """Calculate FID (Fréchet Inception Distance).

    Ref:
    GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash
    Equilibrium. In NeurIPS 2017. <https://arxiv.org/pdf/1706.08500.pdf>

    Args:
        paths (list of two str): Two paths to the real and generated images.
        batch_size (int): Batch size. A reasonable batch size depends on the
            hardware. Default: 50.
        device (str): Device to use, like 'cuda', 'cuda:0' or 'cpu'.
            Default: None (if set to None, it depends the availability).
        dims (int): Dimensionality of Inception features. Default: 2048.

    Returns:
        float: fid result.
    """

    assert len(paths) == 2, ('Two valid image paths should be given, '
                             f'but got {len(paths)} paths')

    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    return fid_score.calculate_fid_given_paths(paths=paths, batch_size=batch_size, device=device, dims=dims)


def lfd(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate LFD (Log Frequency Distance).

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LFD calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lfd result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    freq1 = np.fft.fft2(img1)
    freq2 = np.fft.fft2(img2)
    return np.log(np.mean((freq1.real - freq2.real)**2 + (freq1.imag - freq2.imag)**2) + 1.0)


def mse(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate MSE (Mean Squared Error).

    Ref: https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the MSE calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: mse result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    return np.mean((img1 - img2)**2)


def print_and_write_log(message, log_file=None):
    """Print message and write to a log file.

    Args:
        message (str): The message to print out and log.
        log_file (str, optional): Path to the log file. Default: None.
    """
    print(message)
    if log_file is not None:
        with open(log_file, 'a+') as f:
            f.write('%s\n' % message)
