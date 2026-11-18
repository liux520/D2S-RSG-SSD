from datetime import datetime
import os
import math
import numpy as np
import torch
import cv2
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F


def tensor2img(tensor: torch.Tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().detach().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
             (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(
            tensor, nrow=int(math.sqrt(n_img)), normalize=False
        ).numpy()
        img_np = np.transpose(
            img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(
            img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def calculate_psnr_ssim(img1, img2, scale, min_max=(0, 1)):
    img1, img2 = tensor2img(img1, np.uint8, min_max), tensor2img(
        img2, np.uint8, min_max
    )
    img1, img2 = crop_border([img1, img2], scale)
    p = calculate_psnr(img1, img2)
    s = calculate_ssim(img1, img2)
    return p, s


def calculate_psnr(
        img1: torch.Tensor | np.ndarray,
        img2: torch.Tensor | np.ndarray,
) -> float:
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim2d(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """

    :param img1: HxW or CxHxW or NxCxHxW
    :param img2:
    :return:
    """
    img1, img2 = np.squeeze(img1), np.squeeze(img2)
    if img1.ndim == 2:
        return calculate_ssim2d(img1, img2)
    elif img1.ndim == 3:
        ssim = 0
        for i in range(img1.shape[2]):
            ssim += calculate_ssim2d(img1[:, :, i], img2[:, :, i])
        return ssim / img1.shape[2]
    elif img1.ndim == 4:
        ssim = 0
        for i in range(img1.shape[3]):
            ssim += calculate_ssim(img1[:, :, :, i], img2[:, :, :, i])
        return ssim / img1.shape[3]
    else:
        raise ValueError("dimensions error")


def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [
            v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list
        ]
