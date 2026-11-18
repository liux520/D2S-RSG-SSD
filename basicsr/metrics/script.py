import os
from pprint import pprint
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from natsort import os_sorted


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


def test(lq_path, gt_path, suffix_denoise='', suffix_gt='', crop_border=0, test_y=False, mode='BasicSR'):
    # noise_imgs = os_sorted([os.path.join(denoised_path, im) for im in os.listdir(denoised_path)])[:32]
    # gt_imgs = os_sorted([os.path.join(gt_path, im) for im in os.listdir(gt_path)])[:32]
    # lq_imgs = _get_paths_from_images(lq_path, suffix_denoise) if isinstance(lq_path, (tuple, list)) else [lq_path]
    # gt_imgs = _get_paths_from_images(gt_path, suffix_gt) if isinstance(gt_path, (tuple, list)) else [gt_path]
    if not os.path.isfile(lq_path):
        lq_imgs = _get_paths_from_images(lq_path, suffix=suffix_denoise)
    else:
        lq_imgs = lq_path if isinstance(lq_path, (tuple, list)) else [lq_path]

    if not os.path.isfile(gt_path):
        gt_imgs = _get_paths_from_images(gt_path, suffix=suffix_gt)
    else:
        gt_imgs = gt_path if isinstance(gt_path, (tuple, list)) else [lq_path]

    psnr_list, ssim_list = [], []

    for lq, gt in zip(lq_imgs, gt_imgs):
        lq_img = cv2.imread(lq)[:, :, ::-1]
        gt_img = cv2.imread(gt)[:, :, ::-1]

        # skimage
        if mode == 'skimage':
            if test_y:
                lq_img = cv2.cvtColor(lq_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            if crop_border != 0:
                lq_img = lq_img[crop_border:-crop_border, crop_border:-crop_border, ...]
                gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, ...]
            psnr = compare_psnr(lq_img, gt_img)
            ssim = compare_ssim(lq_img, gt_img,
                                data_range=255, gaussian_weights=True, channel_axis=2, use_sample_covariance=False)

        # BasicSR
        elif mode == 'BasicSR':
            psnr = calculate_psnr(lq_img, gt_img, crop_border=crop_border, input_order='HWC', test_y_channel=test_y,
                                  color_order='RGB')
            ssim = calculate_ssim(lq_img, gt_img, crop_border=crop_border, input_order='HWC', test_y_channel=test_y,
                                  color_order='RGB')

        print(f'{os.path.basename(lq)} -> {os.path.basename(gt)}: PSNR:{psnr:.4f} SSIM:{ssim:.4f}')

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        # print(f'{os.path.basename(lq)}: PSNR:{psnr} SSIM:{ssim}')

    mean_psnr, mean_ssim = np.mean(psnr_list), np.mean(ssim_list)
    print(f'Avg PSNR:{mean_psnr} Avg SSIM:{mean_ssim}')
    return mean_psnr, mean_ssim

if __name__ == '__main__':

    test(
        lq_path=r'',
        gt_path=r'',
        suffix_denoise='',  # '_DN_',
        suffix_gt='',  # '_CL.png',
        crop_border=0,
        test_y=False,
        mode='skimage'
    )
