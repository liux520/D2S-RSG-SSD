import os, cv2, argparse
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
from demo.test_loader import TestH5, TestFolder
from basicsr.archs.deamnet_arch import Deam
from torchvision.utils import save_image


class TestClass(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.stabe_mask = StableMask()
        self.random_mask = MultiScaleMask(2)

    @torch.inference_mode()
    def stage_1(self, model, noisy, gt, pd_factor, basename):
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
    def stage_2(self, model, noisy, coarse, gt, pd_factor, r3_num, r3_factor, basename, p = 0):
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
    def pipeline(self, stage1_model, stage2_model, dataloder,
                 if_refine=False, pd_factor=2, r3_num=8, r3_factor=0.16,
                 print_=False, save=False, save_path='', save_suffix=''):
        if save:
            os.makedirs(save_path, exist_ok=True)

        psnr, ssim = AverageMeter(), AverageMeter()

        stage1_model.to(self.device).eval()
        stage2_model.to(self.device).eval()

        for i, data in enumerate(tqdm(dataloder)):
            lq_tensor = data[0].to(self.device)
            gt_tensor = data[1].to(self.device)
            basename = data[2]

            # # output = self.stage_1(stage1_model, lq_tensor, gt_tensor, pd_factor, basename[0])
            # # tile process
            # output = tile_process_s1(
            #     tile_size=512,
            #     tile_overlap=128,
            #     stage_s1_func=self.stage_1,
            #     model_s1=stage1_model,
            #     lq_latent=lq_tensor,
            #     gt_latent=gt_tensor,
            #     pd_factor=pd_factor,
            #     basename=basename[0],
            #     device=self.device
            # )
            #
            # if if_refine:
            #     # output = self.stage_2(stage2_model, lq_tensor, output, gt_tensor, pd_factor, r3_num, r3_factor, basename[0])
            #     # tile process
            #     output = tile_process_s2(
            #         tile_size=512,
            #         tile_overlap=128,
            #         stage_s2_func=self.stage_2,
            #         model_s2=stage2_model,
            #         lq_latent=lq_tensor,
            #         s1_output=output,
            #         gt_latent=gt_tensor,
            #         pd_factor=pd_factor,
            #         r3_num=r3_num,
            #         r3_factor=r3_factor,
            #         basename=basename[0],
            #         device=self.device
            #     )
            output = tile_process_s1_s2_full_pipeline(
                tile_size=512,
                tile_overlap=128,
                stage_s1_func=self.stage_1,
                model_s1=stage1_model,
                stage_s2_func=self.stage_2,
                model_s2=stage2_model,
                lq_latent=lq_tensor,
                gt_latent=gt_tensor,
                pd_factor=pd_factor,
                r3_num=r3_num,
                r3_factor=r3_factor,
                basename=basename[0],
                device=self.device
            )


            psnr_temp, ssim_temp = calculate_psnr_ssim(output, gt_tensor, scale=0, min_max=(0., 1.))
            psnr.update(psnr_temp)
            ssim.update(ssim_temp)

            if print_:
                print(f'Processing {i}: {basename[0]} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

            if save:
                output_copy = tensor2uint8(output)
                cv2.imwrite(os.path.join(save_path, f'{save_suffix}_{basename[0]}'), output_copy[:, :, ::-1])  # {base}{ext}
        avg_psnr = psnr.avg
        avg_ssim = ssim.avg
        print(f'PSNR/SSIM:{avg_psnr:.6f}/{avg_ssim:.6f} len:{psnr.count}')

        return avg_psnr, avg_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default='Real')
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--save_path', type=str, default=r'results')
    args = parser.parse_args()

    DATASETS = {
        'Real': [r'real/noisy', r'real/noisy']
    }
    ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    CKPT_ROOT = os.path.join(ROOT, 'ckpts')
    DATA_ROOT = os.path.join(ROOT, 'datasets')
    SAVE_ROOT = os.path.join(ROOT, args.save_path, args.dataset)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    noisy_dir, clean_dir = DATASETS[args.dataset]
    noisy_dir, clean_dir = os.path.join(DATA_ROOT, noisy_dir), os.path.join(DATA_ROOT, clean_dir)
    dataset_val = TestFolder(noisy_dir, clean_dir)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    ckpt = f"D2S-RSG-SSD(E)-PolyU.pth"
    model1 = Deam(Isreal=True, scale=16)
    model2 = Deam(Isreal=True, scale=16)
    model1 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model1, key='',
                  delete_module=True, delete_str=['dn.branches.0.'], print_=True)
    model2 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model2, key='',
                  delete_module=True, delete_str=['dn1.branches.0.'], print_=True)

    Inference = TestClass()
    psnr, ssim = Inference.pipeline(
        stage1_model=model1,
        stage2_model=model2,
        dataloder=dataloader_val,
        if_refine=True,
        pd_factor=2,   # SIDD/DND=2, PolyU=1
        r3_num=8,
        r3_factor=0.16,
        print_=False,
        save=args.save,
        save_path=SAVE_ROOT,
        save_suffix='D2S-RSG-E'
    )
