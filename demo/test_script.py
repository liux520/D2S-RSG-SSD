import os, cv2
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from demo.test_utils import pixel_shuffle_func, pixel_unshuffle_func, _get_paths_from_images, AverageMeter, \
    uint2tensor, tensor2uint8, r3, load, pixel_unshuffle, pixel_shuffle
from basicsr.metrics.psnr_ssim_compute import calculate_psnr_ssim_auto
from demo.test_loader import TestFolder, SIDD_val
from basicsr.models.DRSR_utils.mask import StableMask, MultiScaleMask
import torch.nn.functional as F
from basicsr.metrics.compute_metrics import calculate_psnr_ssim
from demo.test_utils import get_time, init_random_seed, str2bool
from demo.tile import tile_process_s1, tile_process_s2, tile_process_s1_s2_full_pipeline

from torchvision.utils import save_image
from demo.test_loader import TestH5, TestFolder
from basicsr.archs.deamnet_arch import Deam
import argparse


class TestClass(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.stabe_mask = StableMask()
        self.random_mask = MultiScaleMask(2)

    @torch.inference_mode()
    def stage_1(self, model, noisy, gt, pd_factor):
        pd = pixel_unshuffle_func(noisy, pd_factor)  # y.*M-->PD  # default=2
        pd_mask, mask = self.stabe_mask(pd)
        pd1 = model(pd_mask[0])  # PDdown-->-y.M->model
        pd2 = model(pd_mask[1])
        test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
        test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
        test_out1 = pixel_shuffle_func(test_out1, pd_factor)  # y.*M-->PDdown-->model-->PDup  # default=2
        test_out2 = pixel_shuffle_func(test_out2, pd_factor)  # default=2
        output_1 = test_out1 + test_out2

        return output_1

    @torch.inference_mode()
    def stage_2(self, model, noisy, coarse, gt, pd_factor, r3_num, r3_factor, p = 0):
        b, c, h, w = coarse.shape
        temp_input = coarse.expand(r3_num, -1, -1, -1)
        lq_tensor = noisy.expand(r3_num, -1, -1, -1)
        indices = torch.zeros(r3_num, c, h, w, dtype=torch.bool, device=lq_tensor.device)
        for t in range(r3_num):
            indices[t] = (torch.rand(1, h, w) < r3_factor).repeat(3, 1, 1)
            temp_input = temp_input.clone()
            temp_input[indices] = lq_tensor[indices]
            temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
        if p == 0:
            pd = pixel_unshuffle_func(temp_input, 1)  # y.*M-->PD
            pd_mask, mask = self.random_mask(pd)
            pd1 = model(pd_mask[0])  # PDdown-->-y.M->model
            pd2 = model(pd_mask[1])
            test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
            test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
            test_out1 = pixel_shuffle_func(test_out1, 1)  # y.*M-->PDdown-->model-->PDup
            test_out2 = pixel_shuffle_func(test_out2, 1)
            output_2 = test_out1 + test_out2
        else:
            pd = pixel_unshuffle_func(temp_input[:, :, p:-p, p:-p], 1)  # y.*M-->PD
            pd_mask, mask = self.random_mask(pd)
            pd1 = model(pd_mask[0])  # PDdown-->-y.M->model
            pd2 = model(pd_mask[1])
            test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
            test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
            test_out1 = pixel_shuffle_func(test_out1, 1)  # y.*M-->PDdown-->model-->PDup
            test_out2 = pixel_shuffle_func(test_out2, 1)
            output_2 = test_out1 + test_out2
        output = torch.mean(output_2, dim=0).unsqueeze(0)

        return output

    @torch.inference_mode()
    def pipeline(self, stage1_model, stage2_model, dataloader,
                 if_refine=False, pd_factor=2, r3_num=8, r3_factor=0.16,
                 print_=False, save=False, save_path='', save_suffix=''):
        if save:
            os.makedirs(save_path, exist_ok=True)

        psnr, ssim = AverageMeter(), AverageMeter()

        stage1_model.to(self.device).eval()
        stage2_model.to(self.device).eval()

        for i, data in enumerate(tqdm(dataloader)):
            lq_tensor = data[0].to(self.device)
            gt_tensor = data[1].to(self.device)
            base, ext = os.path.splitext(data[2][0])

            output = self.stage_1(stage1_model, lq_tensor, gt_tensor, pd_factor)
            if if_refine:
                output = self.stage_2(stage2_model, lq_tensor, output, gt_tensor, pd_factor, r3_num, r3_factor)

            psnr_temp, ssim_temp = calculate_psnr_ssim(output, gt_tensor, scale=0, min_max=(0., 1.))
            psnr.update(psnr_temp)
            ssim.update(ssim_temp)

            if print_:
                print(f'Processing {i}: {base} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

            if save:
                output_copy = tensor2uint8(output)
                cv2.imwrite(os.path.join(save_path, f'{base}_{save_suffix}{ext}'), output_copy[:, :, ::-1])  # {base}{ext}
        avg_psnr = psnr.avg
        avg_ssim = ssim.avg
        print(f'PSNR/SSIM:{avg_psnr:.6f}/{avg_ssim:.6f} len:{psnr.count}')

        return avg_psnr, avg_ssim

    @torch.inference_mode()
    def distillation(self, distillation_model, dataloader,
                     print_=False, save=False, save_path='', save_suffix=''):
        if save:
            os.makedirs(save_path, exist_ok=True)

        psnr, ssim = AverageMeter(), AverageMeter()

        distillation_model.to(self.device).eval()

        for i, data in enumerate(tqdm(dataloader)):
            lq_tensor = data[0].to(self.device)
            gt_tensor = data[1].to(self.device)
            base, ext = os.path.splitext(data[2][0])

            output = distillation_model(lq_tensor)

            psnr_temp, ssim_temp = calculate_psnr_ssim(output, gt_tensor, scale=0, min_max=(0., 1.))
            psnr.update(psnr_temp)
            ssim.update(ssim_temp)

            if print_:
                print(f'Processing {i}: {base} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

            if save:
                output_copy = tensor2uint8(output)
                cv2.imwrite(os.path.join(save_path, f'{base}_{save_suffix}{ext}'), output_copy[:, :, ::-1])  # {base}{ext}
        avg_psnr = psnr.avg
        avg_ssim = ssim.avg
        print(f'PSNR/SSIM:{avg_psnr:.6f}/{avg_ssim:.6f} len:{psnr.count}')

        return avg_psnr, avg_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='PolyU')
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], default='stage3')
    parser.add_argument('--save', type=str2bool, default=False)
    parser.add_argument('--save_path', type=str, default=r'results')
    args = parser.parse_args()

    ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    CKPT_ROOT = os.path.join(ROOT, 'ckpts')
    DATA_ROOT = r'G:\PhD\Paper\LAB\WEIGUANG\TPAMI\Compare\datasets'  # os.path.join(ROOT, 'datasets')
    SAVE_ROOT = os.path.join(ROOT, args.save_path, args.dataset)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    DATASETS = {
        'SIDD': [r'SIDD/noisy', r'SIDD/gt'],
        # 'SIDDBenchmark': [r'SIDDBenchamrk/noisy', r'SIDDBenchamrk/noisy'],
        'PolyU': [r'PolyU/noisy', r'PolyU/gt'],
        # 'Real': [r'real/noisy', r'real/noisy'],
        # 'DND': [r'DND/noisy', r'DND/noisy'],
    }
    noisy_dir, clean_dir = DATASETS[args.dataset]
    noisy_dir, clean_dir = os.path.join(DATA_ROOT, noisy_dir), os.path.join(DATA_ROOT, clean_dir)
    dataset_val = TestFolder(noisy_dir, clean_dir)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    if args.stage == 'stage1':
        ckpt = "D2S-RSG-SSD.pth"
        model = Deam(Isreal=True, scale=16)
        model = load(path=os.path.join(CKPT_ROOT, ckpt), model=model, key='', delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model1 = model2 = model
    elif args.stage == 'stage2':
        ckpt = f"D2S-RSG-SSD(E)-{args.dataset}.pth" if not args.dataset == 'Real' else f"D2S-RSG-SSD(E)-PolyU.pth"
        model1 = Deam(Isreal=True, scale=16)
        model2 = Deam(Isreal=True, scale=16)
        model1 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model1, key='',
                      delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model2 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model2, key='',
                      delete_module=True, delete_str=['dn1.branches.0.'], print_=True)
    elif args.stage == 'stage3':
        ckpt = f"D2S-RSG-SSD(D)-{args.dataset}.pth"
        model_distillation = Deam(Isreal=True, scale=16)
        model_distillation = load(path=os.path.join(CKPT_ROOT, ckpt), model=model_distillation,
                                  key='params', delete_module=False, print_=True)

    Inference = TestClass()

    if args.stage == 'stage1' or args.stage == 'stage2':
        # self-supervised
        psnr, ssim = Inference.pipeline(
            stage1_model=model1,
            stage2_model=model2,
            dataloader=dataloader_val,
            if_refine=False if args.stage == 'stage1' else True,
            pd_factor=1 if args.dataset == 'PolyU' else 2,   # SIDD/DND=2, PolyU=1
            r3_num=8,
            r3_factor=0.16,
            print_=False,
            save=args.save,
            save_path=SAVE_ROOT,
            save_suffix='D2S-RSG' if args.stage == 'stage1' else 'D2S-RSG-E',
        )
    else:
        # self-distillation
        psnr, ssim = Inference.distillation(
            distillation_model=model_distillation,
            dataloader=dataloader_val,
            print_=False,
            save=args.save,
            save_path=SAVE_ROOT,
            save_suffix='D2S-RSG-D'
        )


