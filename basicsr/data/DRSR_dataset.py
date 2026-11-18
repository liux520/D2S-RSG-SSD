from os import path as osp

import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, paired_paths_from_folder_oswalk
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.data.transforms import augment, paired_random_crop
import numpy as np
import torch, random


@DATASET_REGISTRY.register()
class SingleImageDRSRDataset(data.Dataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleImageDRSRDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)

            if self.opt.get('truncation', False):
                # random.shuffle(self.paths)
                num = int(self.opt['truncation'])
                self.paths = self.paths[:num]

        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.lq_folder, line.rstrip().split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

        if 'dataroot_light_noise' in self.opt and self.opt['dataroot_light_noise'] != '':
            self.light_paths = sorted(list(scandir(self.opt['dataroot_light_noise'], full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if 'dataroot_light_noise' in self.opt and self.opt['dataroot_light_noise'] != '':
            light_index = index % len(self.light_paths)
            img_light = cv2.imread(self.light_paths[light_index])[:, :, ::-1].astype(np.float32) / 255.

        if self.opt['phase'] == 'train':
            # random crop
            img_lq = self._random_crop_patch(img_lq, self.opt['gt_size'])
            # flip, rotation
            img_lq = augment([img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            if 'dataroot_light_noise' in self.opt and self.opt['dataroot_light_noise'] != '':
                img_light = self._random_crop_patch(img_light, self.opt['gt_size'])
                img_light = augment([img_light], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        if 'dataroot_light_noise' in self.opt and self.opt['dataroot_light_noise'] != '':
            img_light = img2tensor(img_light, bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        if 'dataroot_light_noise' in self.opt and self.opt['dataroot_light_noise'] != '':
            return {'lq': img_lq, 'lq_path': lq_path, 'lq_light': img_light}
        else:
            return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)

    def _random_crop_patch(self, img, patch_size):
        if type(img) == np.ndarray:
            h, w, c = img.shape
            h_space = random.randint(0, h - patch_size)
            w_space = random.randint(0, w - patch_size)
            return img[h_space:h_space + patch_size, w_space:w_space + patch_size, :]
        elif type(img) == torch.Tensor:
            b, c, h, w = img.shape
            h_space = random.randint(0, h - patch_size)
            w_space = random.randint(0, w - patch_size)
            return img[:, :, h_space:h_space + patch_size, w_space:w_space + patch_size]
        else:
            w, h = img.size
            h_space = random.randint(0, h - patch_size)
            w_space = random.randint(0, w - patch_size)
            return img.crop((w_space, h_space, w_space + patch_size, h_space + patch_size))
