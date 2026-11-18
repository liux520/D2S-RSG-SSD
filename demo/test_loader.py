import cv2
import os
import torch, random
from torch.utils.data import Dataset
import h5py
import scipy
import numpy as np
from skimage.util.dtype import img_as_float32 as img_as_float
from demo.test_utils import _get_paths_from_images


class TestH5(Dataset):
    def __init__(self, h5_path):
        super(TestH5).__init__()

        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2 / 2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)).astype(np.float32) / 255.).float()
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)).astype(np.float32) / 255.).float()

        return im_noisy, im_gt

    def crop_patch(self, imgs_sets, patch_size):
        H, W, C2 = imgs_sets.shape
        C = int(C2 / 2)
        ind_H = random.randint(0, H - patch_size)
        ind_W = random.randint(0, W - patch_size)
        im_noisy = np.array(imgs_sets[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size, :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size, C:])
        return im_gt, im_noisy


class TestFolder(Dataset):
    def __init__(self, lq_path, hq_path):
        super(TestFolder).__init__()

        if os.path.isdir(lq_path):
            self.hq_imgs = _get_paths_from_images(hq_path)
            self.lq_imgs = _get_paths_from_images(lq_path)
        else:
            self.hq_imgs = [hq_path]
            self.lq_imgs = [lq_path]

    def __len__(self):
        return len(self.hq_imgs)

    def __getitem__(self, item):
        hq_p = self.hq_imgs[item]
        lq_p = self.lq_imgs[item]

        hq = cv2.imread(hq_p)[:, :, ::-1].astype(np.float32) / 255.  #, cv2.IMREAD_COLOR
        lq = cv2.imread(lq_p)[:, :, ::-1].astype(np.float32) / 255.

        return self.np2tensor(lq), self.np2tensor(hq), os.path.basename(hq_p)

    def np2tensor(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


class SIDD_val(Dataset):
    def __init__(self, mat_path=r'E:\Dataset\Restoration\Denoise\Real\SIDD\val', len=1280):
        super().__init__()
        sidd_val_dir = mat_path
        assert os.path.exists(sidd_val_dir), 'There is no dataset %s' % sidd_val_dir

        clean_mat_file_path = os.path.join(sidd_val_dir, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(sidd_val_dir, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(
            clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(
            noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, data_idx):
        img_id = data_idx // 32
        patch_id = data_idx % 32

        clean_img = self.clean_patches[img_id, patch_id, :].astype(float)
        noisy_img = self.noisy_patches[img_id, patch_id, :].astype(float)

        clean_img = torch.from_numpy(clean_img.transpose(2, 0, 1))
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1))

        return {'clean': clean_img, 'noisy': noisy_img, 'clean_key': f'{img_id}_{patch_id}.png'}


def padding(img_lq, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    return img_lq


if __name__ == '__main__':
    dataset = TestH5(h5_path=r'E:\Dataset\Restoration\Denoise\Real\SIDD\train\small_imgs_test.hdf5')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for im, gt in dataloader:
        print(im.shape)
        print(gt.shape)
