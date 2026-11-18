# rsg_fast.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class StableMask(nn.Module):
    """向量化实现的 StableMask：返回两个被掩蔽的图像和对应掩码（bool tensors）"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (b, c, h, w)
        b, c, h, w = x.shape
        # checkerboard mask: True 表示属于 mask1 (we'll call it mask_A)
        mask_A = torch.zeros((b, 1, h, w), dtype=torch.bool, device=x.device)
        mask_A[:, :, ::2, ::2] = True
        mask_A[:, :, 1::2, 1::2] = True
        mask_B = ~mask_A
        # expand to channels
        mask_Ac = mask_A.expand(-1, c, -1, -1)
        mask_Bc = mask_B.expand(-1, c, -1, -1)
        # masked images (float)
        img_A = x * mask_Ac.float()
        img_B = x * mask_Bc.float()
        # return stacked masked images and list of masks in expected order [mask_A, mask_B]
        return torch.stack([img_A, img_B], dim=0), [mask_Ac, mask_Bc]


def generate_mask_pair(img, p):
    n, c, h, w = img.shape
    h2 = h // p
    w2 = w // p
    total = n * h2 * w2
    device = img.device
    rnd = torch.rand(total, p * p, device=device)
    perm = rnd.argsort(dim=1)  # shape (total, p*p)
    perm_idx = perm.transpose(0, 1).contiguous()  # shape (p*p, total)
    # TODO: 直接生成 (p*p, total) index matrix ?
    return perm_idx  # dtype: int64, values in [0..p*p-1]


def generate_subimages_new2(img, perm_idx, p=2):
    n, c, h, w = img.shape
    h2 = h // p
    w2 = w // p
    pp = p * p
    device = img.device

    patches = img.view(n, c, h2, p, w2, p) \
                 .permute(3, 5, 0, 1, 2, 4).contiguous() \
                 .view(pp, n, c, h2, w2)  # (pp, n, c, h2, w2)

    total = n * h2 * w2
    patches_flat = patches.view(pp, n, c, h2 * w2)       # (pp, n, c, h2*w2)
    patches_flat = patches_flat.permute(0, 2, 1, 3).contiguous()  # (pp, c, n, total)
    patches_flat = patches_flat.view(pp, c, total)       # (pp, c, total)

    idx = perm_idx.unsqueeze(1).expand(-1, c, -1)  # (pp, total) -> (pp, c, total)
    gathered_flat = torch.gather(patches_flat, 0, idx)  # (pp, c, total)

    gathered = gathered_flat.view(pp, c, n, h2, w2).permute(0, 2, 1, 3, 4).contiguous()
    return perm_idx, gathered


def subimages_reverse_new(subimages, p=2):
    pp, n, c, h2, w2 = subimages.shape
    assert pp == p * p
    patches_pp = subimages.view(p, p, n, c, h2, w2).permute(2, 3, 4, 0, 5, 1).contiguous()
    out = patches_pp.view(n, c, h2 * p, w2 * p)
    return out


def subimages_reverse_new_with_perm(subimages, perm_idx, p=2):
    pp, n, c, h2, w2 = subimages.shape
    assert pp == p * p
    device = subimages.device
    total = n * h2 * w2
    assert perm_idx.shape == (pp, total)

    gathered_flat = subimages.permute(0, 2, 1, 3, 4).contiguous().view(pp, c, total)  # (pp, c, total)

    idx = perm_idx.unsqueeze(1).expand(-1, c, -1)  # (pp, c, total)
    patches_flat = torch.zeros_like(gathered_flat)
    patches_flat.scatter_(0, idx, gathered_flat)

    patches = patches_flat.view(pp, c, n, h2, w2).permute(0, 2, 1, 3, 4).contiguous()  # (pp, n, c, h2, w2)
    patches_pp = patches.view(p, p, n, c, h2, w2).permute(2, 3, 4, 0, 5, 1).contiguous()  # (n, c, h2, p, w2, p)
    out = patches_pp.view(n, c, h2 * p, w2 * p)
    return out


def RSG(lq, pd_factor, mask_module):
    """
    返回 noisy_sub, noisy_sub1, noisy_sub2, placeholders for denoised, mask_sub(二值标记),
    location_map (perm_idx)
    """
    n, c, h, w = lq.shape
    # prepare location permutation index
    perm_idx = generate_mask_pair(lq, pd_factor)  # shape (p*p, total)
    x_mask, masks = mask_module(lq)  # returns stacked masked images and list of masks
    # generate subimages
    _, noisy_sub = generate_subimages_new2(lq, perm_idx, p=pd_factor)
    _, noisy_sub1 = generate_subimages_new2(x_mask[0], perm_idx, p=pd_factor)
    _, noisy_sub2 = generate_subimages_new2(x_mask[1], perm_idx, p=pd_factor)
    # mask_sub we produce as binary indicating which positions belong to mask (for the
    # two masks A/B). We'll create mask_sub with same spatial layout as subimages: (p*p, n, 1, h2, w2)
    # To get mask_sub: take one of masks (mask_A expanded) and extract patches deterministically
    # We'll reuse generate_subimages_new logic: treat mask image as float
    _, mask_sub = generate_subimages_new2(masks[0].float(), perm_idx, p=pd_factor)
    # mask_sub shape (p*p, n, 1, h2, w2)
    noisy_sub1_denoised = torch.zeros_like(noisy_sub1)
    noisy_sub2_denoised = torch.zeros_like(noisy_sub2)
    return noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, perm_idx


def forward_fast_last(lq, net_g1, pd_factor_train=5):
    noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, perm_idx = (
        RSG(lq, pd_factor_train, StableMask()))
    pp = pd_factor_train * pd_factor_train
    # for each subimage call the denoiser; we keep the loop here because invocation of net_g1 is required per subimage
    # but the subimages themselves are in contiguous GPU memory and small loop overhead is fine.

    # for j in range(pp):
    #     noisy_sub1_denoised[j] = net_g1(noisy_sub1[j])
    #     noisy_sub2_denoised[j] = net_g1(noisy_sub2[j])

    pp, n, c, h2, w2 = noisy_sub1.shape
    noisy_sub1_denoised = net_g1(noisy_sub1.view(pp * n, c, h2, w2)).view(pp, n, c, h2, w2)
    noisy_sub2_denoised = net_g1(noisy_sub2.view(pp * n, c, h2, w2)).view(pp, n, c, h2, w2)

    # Compose output: (use mask_sub to select between denoised results)
    # mask_sub is float (0/1)
    # combine per-subimage with mask_sub
    # part1 = subimages_reverse_new((1.0 - mask_sub) * noisy_sub1_denoised, p=pd_factor_train)
    # part2 = subimages_reverse_new(mask_sub * noisy_sub2_denoised, p=pd_factor_train)
    part1 = subimages_reverse_new_with_perm((1.0 - mask_sub) * noisy_sub1_denoised, p=pd_factor_train, perm_idx=perm_idx)
    part2 = subimages_reverse_new_with_perm(mask_sub * noisy_sub2_denoised, p=pd_factor_train, perm_idx=perm_idx)
    output = part1 + part2
    return output, part1, part2, noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, perm_idx


def forward_fast(lq, net_g1, pd_factor_train=5, device='cuda'):
    lq = lq.to(device)
    noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, perm_idx = RSG(lq, pd_factor_train, StableMask())
    pp = pd_factor_train * pd_factor_train
    # for each subimage call the denoiser; we keep the loop here because invocation of net_g1 is required per subimage
    # but the subimages themselves are in contiguous GPU memory and small loop overhead is fine.
    # for j in range(pp):
    #     noisy_sub1_denoised[j] = net_g1(noisy_sub1[j])
    #     noisy_sub2_denoised[j] = net_g1(noisy_sub2[j])

    pp, n, c, h2, w2 = noisy_sub1.shape
    noisy_sub1_denoised = net_g1(noisy_sub1.view(pp * n, c, h2, w2)).view(pp, n, c, h2, w2)
    noisy_sub2_denoised = net_g1(noisy_sub2.view(pp * n, c, h2, w2)).view(pp, n, c, h2, w2)

    # Compose output: (use mask_sub to select between denoised results)
    # mask_sub is float (0/1)
    # combine per-subimage with mask_sub
    part1 = subimages_reverse_new((1.0 - mask_sub) * noisy_sub1_denoised, p=pd_factor_train)
    part2 = subimages_reverse_new(mask_sub * noisy_sub2_denoised, p=pd_factor_train)
    output = part1 + part2
    return output, part1, part2, noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, perm_idx


"""
# 假设 noisy_sub1 shape: (pp, n, c, h2, w2)
pp, n, c, h2, w2 = noisy_sub1.shape
# 合并成 batch: (pp*n, c, h2, w2)
batch1 = noisy_sub1.view(pp * n, c, h2, w2)
denoised_batch1 = net_g1(batch1)  # single forward
# reshape 回 (pp, n, c, h2, w2)
noisy_sub1_denoised = denoised_batch1.view(pp, n, c, h2, w2)
"""

