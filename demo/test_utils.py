import os, cv2
import numpy as np
import torch
from natsort import os_sorted
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import random
import math
import time, argparse
# import logging
# from basicsr.utils import (get_root_logger, get_time_str)
# logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load(path, model, key='state_dict', delete_module=False, delete_str=['module.', ], print_=True):
    from collections import OrderedDict

    def delete_state_module(weights, delete_str='module.'):
        weights_dict = {}
        for k, v in weights.items():
            # new_k = k.replace('module.', '') if 'module' in k else k
            new_k = k.replace(delete_str, '') if delete_str in k else k
            weights_dict[new_k] = v
        return weights_dict

    ckpt = torch.load(path, map_location=lambda storage, loc: storage)  # ['state_dict']
    ckpt = ckpt.get(key, ckpt)
    # ckpt = ckpt[key] if key != '' else ckpt

    if delete_module:
        for d_s in delete_str:
            ckpt = delete_state_module(ckpt, d_s)
    # for k in ckpt.keys():
    #     print(k)

    overlap = OrderedDict()
    for key, value in ckpt.items():
        if key in model.state_dict().keys() and ckpt[key].shape == model.state_dict()[key].shape:
            overlap[key] = value
        # else:
        #     print(f'Failed load: ckpt: {key}-{ckpt[key].shape} | model: {key}-{model.state_dict()[key].shape}')

    if print_:
        print(f'{(len(overlap) * 1.0 / len(ckpt) * 100):.4f}% weights are loaded!', end='\t')
        print(f'{(len(overlap) * 1.0 / len(model.state_dict()) * 100):.4f}% params are inited!')

    try:
        model.load_state_dict(overlap, strict=False)
        if print_: print(f'Loading weights from {path}.')
    except RuntimeError as e:
        print(f'RuntimeError: {e}')

    return model


def r3(x: torch.Tensor, denoised: torch.Tensor, net: nn.Module, r3_factor: float = 0.16, r3_num: int = 8, p: int = 0) -> torch.Tensor:
    """random replacement Refinement with ratio r3_factor

    Note:
        This module is only used in eval, not in train. val will take r3_num times longer than train.
    Args:
        x(torch.Tensor): input tensor.BCHW,B=1
        net(nn.Module): model to eval
        r3_factor (float, optional): the ratio of radnom replace. Defaults to 0.16.
        r3_num (int, optional): the number of r3 times. Defaults to 8.
    Output: BCHW
    """
    if r3_factor < 0:
        return denoised
    else:
        b, c, h, w = x.shape
        temp_input = denoised.expand(r3_num, -1, -1, -1)
        x = x.expand(r3_num, -1, -1, -1)
        indices = torch.zeros(r3_num, c, h, w, dtype=torch.bool, device=x.device)
        for t in range(r3_num):
            indices[t] = (torch.rand(1, h, w) < r3_factor).repeat(3, 1, 1)
        temp_input = temp_input.clone()
        temp_input[indices] = x[indices]
        temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
        with torch.no_grad():
            if p == 0:
                denoised = net(temp_input)
            else:
                denoised = net(temp_input)[:, :, p:-p, p:-p]
        return torch.mean(denoised, dim=0).unsqueeze(0)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
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


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pixel_unshuffle(input, factor):
    """
    (n, c, h, w) ===> (n*factor^2, c, h/factor, w/factor)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // factor
    out_width = in_width // factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, factor,
        out_width, factor)

    batch_size *= factor ** 2
    unshuffle_out = input_view.permute(0, 3, 5, 1, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


def pixel_shuffle(input, factor):
    """
    (n*factor^2, c, h/factor, w/factor) ===> (n, c, h, w)
    """
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height * factor
    out_width = in_width * factor

    batch_size /= factor ** 2
    batch_size = int(batch_size)
    input_view = input.contiguous().view(
        batch_size, factor, factor, channels, in_height,
        in_width)

    unshuffle_out = input_view.permute(0, 3, 4, 1, 5, 2).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)



def single2uint(image):
    '''Numpy:(0,1) (h,w,c) float32 -> Numpy:(0,255) (h,w,c) uint8'''
    return np.uint8((image.clip(0., 1.) * 255.).round())


def uint2single(image):
    '''Numpy:(0,255) (h,w,c) uint8 -> Numpy:(0,1) (h,w,c) float32'''
    return np.float32(image / 255.)


def np2tensor(imgs, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img ,float32):
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, float32) for img in imgs]
    else:
        return _totensor(imgs, float32)


def tensor2np(imgs):
    def _tonp(img):
        img = img.data.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.float32)
        img = np.uint8((img.clip(0., 1.) * 255.).round())
        return img

    if isinstance(imgs, list):
        return [_tonp(img) for img in imgs]
    else:
        return _tonp(imgs)


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        img *= std
        img += mean
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        mean = torch.as_tensor(mean)[None,:, None, None]
        std = torch.as_tensor(std)[None,:, None, None]
        img = img.mul_(std).add_(mean)
    return img


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        img /= 255.
        img -= mean
        img /= std
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        mean = torch.as_tensor(mean)[None, :, None, None]
        std = torch.as_tensor(std)[None, :, None, None]
        img = img.sub_(mean).div_(std)
    return img


def _random_crop_patch(img, patch_size):
    if type(img) == np.ndarray:
        h,w,c=img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[h_space:h_space + patch_size, w_space:w_space + patch_size, :]
    elif type(img) == torch.Tensor:
        b, c, h, w = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[:, :, h_space:h_space + patch_size, w_space:w_space + patch_size]
    else:
        w,h=img.size
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img.crop((w_space,h_space,w_space+patch_size,h_space+patch_size))


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)

    return img, mask


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def subimage(path, crop_size, step, thresh_size, save_folder, compression_level=3):
    """Worker for each process.

    Args:
        path (str): Image path.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """

    img_name, extension = os.path.splitext(os.path.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                os.path.join(save_folder, f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    process_info = f'Processing {img_name} ...'
    return process_info


def patch_test(model, lr, scale=4, tile=320, tile_overlap=24):
    b, c, h, w = lr.size()
    tile = min(tile, h, w)

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * scale, w * scale).type_as(lr)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = lr[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            if isinstance(out_patch, list):
                out_patch = out_patch[-1]
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                out_patch)
            W[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                out_patch_mask)
    res = E.div_(W)
    output = res  # [:, :, :h * 4, :w * 4]
    return output


# WSDFormer
def splitimage(imgtensor, crop_size=128, overlap_size=20):  # 128
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


# https://github.com/XLearning-SCU/2023-CVPR-CODE/blob/main/utils.py
def image2patch(image, patch_size, step):
    '''convert image to patches'''
    assert len(image.size()) == 4 and image.size(0) == 1
    H, W = image.size()[-2:]
    h_space = list(range(0, H - patch_size, step))
    h_space.append(H - patch_size)
    w_space = list(range(0, W - patch_size, step))
    w_space.append(W - patch_size)
    patches = [
        image[..., h:h + patch_size, w:w + patch_size] \
        for h in h_space \
        for w in w_space
    ]
    patches = torch.cat(patches, dim=0)
    return patches


def patch2image(patches, height, width, step):
    image = patch2image_wo_norm(patches, height, width, step)
    norm = patch2image_wo_norm(image2patch(
        torch.ones_like(image), patches.size(-1), step
    ), height, width, step)
    return image / norm


def patch2image_wo_norm(patches, height, width, step):
    '''convert patches to image'''
    # patches = patches[0] if isinstance(patches, (list, tuple)) else patches
    assert len(patches.size()) == 4 and patches.size(-2) == patches.size(-1)
    _, C, patch_size, _ = patches.size()
    image = torch.zeros(1, C, height, width, device=patches.device)
    h_space = list(range(0, height - patch_size, step))
    h_space.append(height - patch_size)
    w_space = list(range(0, width - patch_size, step))
    w_space.append(width - patch_size)
    idx = 0
    for h in h_space:
        for w in w_space:
            image[..., h:h + patch_size, w:w + patch_size] += patches[idx]
            idx += 1
    return image


def augment_imglist(img):
    def _transform(v, op):
        v2np = v.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    x = [img]
    for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])
    return x


def reverse_augment(img_list):
    def _transform(v, op):
        v2np = v.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()
        return ret

    B, C, H, W = img_list[0].shape
    for i in range(len(img_list)):
        if i > 3:
            img_list[i] = _transform(img_list[i], 't')
        if i % 4 > 1:
            img_list[i] = _transform(img_list[i], 'h')
        if (i % 4) % 2 == 1:
            img_list[i] = _transform(img_list[i], 'v')

    img_list = [img.view(1, B, C, H, W) for img in img_list]
    y = [torch.cat(img_list, dim=0).mean(dim=0, keepdim=False)]
    y = y[0]
    return y


def model_x8(inputs, model):
    inputs_list = augment_imglist(inputs)
    with torch.no_grad():
        outputs = [model(input) for input in inputs_list]
    ensembled_outputs = reverse_augment(outputs)
    outputs = ensembled_outputs
    return outputs

def init_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False