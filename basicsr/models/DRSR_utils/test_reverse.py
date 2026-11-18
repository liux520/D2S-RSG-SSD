import time
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from net_func import grid_partition


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

    # image_flatten = image.reshape(-1)
    # for i in range(p ** 2):
    #     mask_ = mask[i].repeat(c)
    #     image_flatten[mask_] = subimage[i].reshape(-1)
    # image = image_flatten.reshape(image.shape)

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


def demask_new(subimages, masks, p=2):
    n, b, c, h, w = subimages.shape
    image = torch.zeros(b * c * h * p * w * p, dtype=subimages.dtype, device=subimages.device)
    for i, mask in enumerate(masks):
        mask = mask.repeat(c)
        image[mask] = subimages[i, :, :, :, :].reshape(-1)
    image = image.reshape(b, c * h * w, p, p)
    image = rearrange(image, 'b (c h w) p1 p2 -> b c (h p1) (w p2)', b=b, c=c, p1=p, p2=p, h=h, w=w)
    return image


if __name__ == '__main__':
    p = 2
    img = torch.randn((16, 3, 250, 250), device='cuda')
    n, c, h, w = img.shape

    for _ in range(50):
        masks = generate_mask_pair(img, p)

        # old codes
        s1 = time.time()

        noisy_sub = torch.zeros([p * p, n, c, h // p, w // p], device='cuda')
        for j in range(p * p):
            noisy_sub[j, ...] = generate_subimages(img, masks[j], p)  # RSG(y)

        s2 = time.time()

        reverse_img = demask(noisy_sub, masks, p)

        s3 = time.time()
        print(f'old codes: [{n},{c},{h},{w}] | generation: {s2-s1}s, reverse: {s3-s2}, result: {torch.equal(img, reverse_img)}')

        # new codes

        e1 = time.time()

        masks, subimages = generate_subimages_new(img, masks, p)

        e2 = time.time()

        reverse_img2 = demask_new(subimages, masks, p)

        e3 = time.time()
        print(f'new codes: [{n},{c},{h},{w}] | generation: {e2 - e1}s, reverse: {e3 - e2}, result: {torch.equal(img, reverse_img2)}')

