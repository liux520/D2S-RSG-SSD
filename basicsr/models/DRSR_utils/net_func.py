import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import images
from torchvision.utils import save_image
from basicsr.models.DRSR_utils.mask import MultiScaleMask, StableMask
import numpy as np
from einops import rearrange

__all__ = [
    'RSG', 'RSG_func',
    'grid_partition', 'window_partition', 'pixel_shuffle_func', 'pixel_unshuffle_func',
    'generate_mask_pair', 'generate_subimages', 'demask',
    'generate_subimages_new', 'subimages_reverse_new',
    'from_maskindexs_subimages_reverse_original_image'
]


def RSG(x, pd_factor, mask):
    n, c, h, w = x.size()
    location_mask = generate_mask_pair(x, pd_factor)  # RSG中的mask

    x_mask, mask = mask(x)  # b, c, h, w

    # print(f'x_mask[0]: {x_mask[0].shape}, mask[1]: {mask[1].shape}, mask[0]: {mask[0].shape}, mask[1]: {mask[1].shape}')

    _, noisy_sub = generate_subimages_new(x, location_mask, p=pd_factor)  # RSG(x): [p*p, b, c, h//p, w//p]
    _, noisy_sub1 = generate_subimages_new(x_mask[0], location_mask, p=pd_factor)  # RSG(y.*M): [p*p, b, c, h//p, w//p]
    _, noisy_sub2 = generate_subimages_new(x_mask[1], location_mask, p=pd_factor)  # RSG(y.*1-M): [p*p, b, c, h//p, w//p]
    # print(f'noisy_sub1: {noisy_sub1.shape}, noisy_sub2: {noisy_sub2.shape}')
    _, mask_sub = generate_subimages_new(mask[0] * 1., location_mask, p=pd_factor)  # RSG(M)
    # print(f'noisy_sub1: {mask_sub.shape}')

    noisy_sub1_denoised = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor], device=x.device)
    noisy_sub2_denoised = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor], device=x.device)

    return noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, location_mask

    # def denoise_pipeline(self, model, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, location_mask, pd_factor):
    #     for num in range(pd_factor * pd_factor):
    #         noisy_sub1_denoised[num, ...] = model(noisy_sub1[num, ...])
    #         noisy_sub2_denoised[num, ...] = model(noisy_sub2[num, ...])
    #
    #     denoised_img = (from_maskindexs_subimages_reverse_original_image(torch.mul(1 - mask_sub, noisy_sub1_denoised), location_mask, pd_factor) +
    #                     from_maskindexs_subimages_reverse_original_image(torch.mul(mask_sub, noisy_sub2_denoised), location_mask, pd_factor))
    #     return denoised_img


def RSG_func(x, pd_factor):
    location_mask = generate_mask_pair(x, pd_factor)  # RSG中的mask
    n, c, h, w = x.size()
    mask = StableMask()
    x_mask, mask = mask(x)
    noisy_sub = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    noisy_sub1 = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    noisy_sub2 = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    m_sub = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    for j in range(pd_factor * pd_factor):
        noisy_sub1[j, ...] = generate_subimages(x_mask[0], location_mask[j], pd_factor)  # RSG(y.*M)
        noisy_sub2[j, ...] = generate_subimages(x_mask[1], location_mask[j], pd_factor)  # RSG(y.*1-M)
        noisy_sub[j, ...] = generate_subimages(x, location_mask[j], pd_factor)  # RSG(y)
        m_sub[j, ...] = generate_subimages(mask[0] * 1., location_mask[j], pd_factor)  # RSG(M)
    # sub_denoised = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).cuda()
    sub_denoised1 = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    sub_denoised2 = torch.zeros([pd_factor * pd_factor, n, c, h // pd_factor, w // pd_factor]).to(x.device)
    return noisy_sub1, noisy_sub2, sub_denoised1, sub_denoised2, noisy_sub, m_sub, location_mask


def grid_partition(x, block_size):
    # (n, c, h, w) ===> (n, c*block_size^2, h/block_size, w/block_size)
    n, c, h, w = x.size()
    unfolded_x = rearrange(
        x, 'b c (h p1) (w p2) -> b (c p1 p2) h w',
        p1=block_size, p2=block_size, h=h // block_size, w=w // block_size, b=n, c=c
    )
    return unfolded_x


def window_partition(img, window_size):
    # (n, c, h, w) ===> (n, c*h//p*w//p, p, p)
    n, c, h, w = img.size()
    img = rearrange(
        img, 'b c (h p1) (w p2) -> b (c h w) p1 p2',
        p1=window_size, p2=window_size, h=h // window_size, w=w // window_size, b=n, c=c
    )
    return img


def pixel_unshuffle_func(x, factor):
    # (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    b, c, h, w = x.size()
    unshuffle_x = rearrange(
        x, 'b c (h p1) (w p2) -> (b p1 p2) c h w',
        b=b, c=c, h=h // factor, w=w // factor, p1=factor, p2=factor
    )
    return unshuffle_x


def pixel_shuffle_func(x, factor):
    # (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    b, c, h, w = x.size()
    shuffle_x = rearrange(
        x, '(b p1 p2) c h w -> b c (h p1) (w p2)',
        b=b // (factor ** 2), c=c, h=h, w=w, p1=factor, p2=factor
    )
    return shuffle_x


def pd_down(x: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = x.shape
    x_down = F.pixel_unshuffle(x, pd_factor)
    out = x_down.view(b, c, pd_factor, pd_factor, h // pd_factor, w // pd_factor).permute(
        0, 2, 3, 1, 4, 5).reshape(b * pd_factor * pd_factor, c, h // pd_factor, w // pd_factor)
    return out


def pd_up(out: torch.Tensor, pd_factor: int = 5, pad: int = 0) -> torch.Tensor:
    b, c, h, w = out.shape
    # Reshape the output tensor to its original shape after pixel unshuffle
    x_down = out.view(b // (pd_factor ** 2), pd_factor, pd_factor, c, h,
                      w).permute(0, 3, 1, 2, 4, 5)
    x_down = x_down.reshape(b // (pd_factor ** 2), c *
                            pd_factor * pd_factor, h, w)
    # Use pixel shuffle to upsample the tensor
    x_up = F.pixel_shuffle(x_down, pd_factor)
    return x_up


# BasicSR
def pixel_unshuffle(x, scale):
    # [b, c, h, w] -> [b, c, r, r, h/p, w/p] -> [b, c*r*r, h/p, w/p]
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

def generate_mask_pair(img, p):
    # prepare masks (N x C x H/p x W/p)
    n, c, h, w = img.shape
    mask = []
    for i in range(p * p):
        mask.append(torch.zeros(size=(n * h // p * w // p * p * p,),
                                dtype=torch.bool,
                                device=img.device))

    rd_pair_idx = np.zeros([n * h // p * w // p, p * p], dtype=np.int64)
    for i in range(n * h // p * w // p):
        array = np.arange(p * p)
        np.random.shuffle(array)
        rd_pair_idx[i, :] = array
    rd_pair_idx = torch.from_numpy(rd_pair_idx).to(img.device)

    rd_pair_idx += torch.arange(start=0,
                                end=n * h // p * w // p * p * p,
                                step=p * p,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    for i in range(p * p):
        mask[i][rd_pair_idx[:, i]] = 1
    return mask


def generate_subimages(img, mask, p):
    n, c, h, w = img.shape
    # print("mask.shape",mask.shape)  # (b,c*p*p,h//p,w//p)
    subimage = torch.zeros(n,
                           c,
                           h // p,
                           w // p,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = grid_partition(img[:, i:i + 1, :, :], block_size=p)  # (b,c,h,w)-->(b,c*p*p, h/p, w/p)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1)
        img_per_channel = img_per_channel.reshape(-1)

        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(n, h // p, w // p, 1).permute(0, 3, 1, 2)

    return subimage


def demask(subimage, mask, p):
    n, b, c, h, w = subimage.shape
    image = torch.zeros(b,
                        c,
                        h * p,
                        w * p,
                        dtype=subimage.dtype,
                        device=subimage.device)
    up = nn.PixelShuffle(p)
    for i in range(c):
        m = image[:, i:i + 1, :, :].reshape(-1)
        for j in range(p ** 2):
            m[mask[j]] = subimage[j, :, i:i + 1, :, :].reshape(-1)
        m = m.reshape(b, h, w, p ** 2).permute(0, 3, 1, 2)
        image[:, i:i + 1, :, :] = up(m)

    return image


def generate_subimages_new(img, masks, p=2):
    n, c, h, w = img.shape

    # Rearrange the image into patches
    patches = rearrange(
        img, 'b c (h p1) (w p2) -> b (c h w) p1 p2',
        p1=p, p2=p, h=h // p, w=w // p, b=n, c=c
    ).reshape(-1)

    # subimages = []
    # for i, mask in enumerate(masks):
    #     mask = mask.repeat(c)
    #     subimage = patches[mask].reshape(n, c, h // p, w // p)
    #     subimages.append(subimage)

    subimages = torch.zeros((len(masks), n, c, h // p, w // p), dtype=img.dtype, device=img.device)
    for i, mask in enumerate(masks):
        mask = mask.repeat(c)
        subimages[i, :, :, :, :] = patches[mask].reshape(n, c, h // p, w // p)

    return masks, subimages


def subimages_reverse_new(subimages, masks, p=2):
    n, b, c, h, w = subimages.shape
    image = torch.zeros(b * c * h * p * w * p, dtype=subimages.dtype, device=subimages.device)
    for i, mask in enumerate(masks):
        mask = mask.repeat(c)
        image[mask] = subimages[i, :, :, :, :].reshape(-1)
    image = image.reshape(b, c * p * p, h, w)
    image = rearrange(image, 'b (c p1 p2) h w -> b c (h p1) (w p2)', b=b, c=c, p1=p, p2=p, h=h, w=w)
    return image


def from_maskindexs_subimages_reverse_original_image(subimages, masks, original_shape, p=2):
    n, c, h, w = original_shape
    # Create an empty tensor for the reconstructed image
    reconstructed = torch.zeros(n, c, h, w, dtype=subimages[0].dtype, device=subimages[0].device)

    # Flatten the masks for use
    flat_masks = [mask.repeat(c) for mask in masks]

    # Get the full spatial resolution shape
    rearranged = rearrange(
        reconstructed, 'b c (h p1) (w p2) -> b (c h w) p1 p2',
        p1=p, p2=p, h=h // p, w=w // p, b=n, c=c
    ).reshape(-1)

    # Reshape subimages back to the original shape
    for i, (subimage, flat_mask) in enumerate(zip(subimages, flat_masks)):
        # Flatten the subimage
        subimage_flat = subimage.reshape(-1)

        # Assign values from the subimage to the appropriate positions
        rearranged[flat_mask] = subimage_flat

    # Restore the rearranged tensor back to the original shape
    reconstructed = rearrange(
        rearranged, '(b c h w p1 p2) -> b c (h p1) (w p2)',
        p1=p, p2=p, h=h // p, w=w // p, b=n, c=c
    )

    return reconstructed


if __name__ == '__main__':
    # n, c, h, w = 1, 3, 8, 8
    # p = 2
    # x = torch.arange(start=0,
    #                  end=n * h // p * w // p * p * p,
    #                  step=p * p)
    # print(x)

    x = torch.randn(1, 3, 250, 250)
    down1 = pd_down(x, 5)
    down2 = pixel_unshuffle_func(x, 5)
    down3 = pixel_unshuffle(x, 5)
    print(down1.shape, down2.shape, down3.shape)
    print(torch.equal(down1, down2))  # https://blog.csdn.net/qq_40206371/article/details/134171507
    print(torch.equal(down1, down3))
    print(torch.equal(down2, down3))