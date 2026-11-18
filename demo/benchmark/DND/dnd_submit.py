import scipy.io as sio
import h5py
import argparse
import os,  torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from demo.test_utils import load, pixel_shuffle_func, pixel_unshuffle_func, str2bool
from basicsr.archs.deamnet_arch import Deam
from basicsr.models.DRSR_utils.mask import StableMask, MultiScaleMask
import torch.nn.functional as F


class TestClass(object):
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.stabe_mask = StableMask()
        self.random_mask = MultiScaleMask(2)

    @torch.inference_mode()
    def stage_1(self, model, noisy, pd_factor):
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
    def stage_2(self, model, noisy, coarse, pd_factor, r3_num, r3_factor, p = 0):
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
    def pipeline(self, stage1_model, stage2_model, lq,
                 if_refine=False, pd_factor=2, r3_num=8, r3_factor=0.16):

        stage1_model.to(self.device).eval()
        stage2_model.to(self.device).eval()

        lq_tensor = torch.from_numpy(np.ascontiguousarray(lq.transpose(2, 0, 1))).unsqueeze(0).to(self.device)

        output = self.stage_1(stage1_model, lq_tensor, pd_factor)
        if if_refine:
            output = self.stage_2(stage2_model, lq_tensor, output, pd_factor, r3_num, r3_factor)

        return output.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def distillation(self, distillation_model, lq):

        distillation_model.to(self.device).eval()
        lq_tensor = torch.from_numpy(np.ascontiguousarray(lq.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        output = distillation_model(lq_tensor)

        return output.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], default='stage2')
    parser.add_argument('--save_path', type=str, default=r'results/submit/DND')
    parser.add_argument('--save_image', type=str2bool, default=False)
    args = parser.parse_args()

    ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
    CKPT_ROOT = os.path.join(ROOT, 'ckpts')
    DATA_ROOT = r'G:\PhD\Paper\LAB\WEIGUANG\TPAMI\Compare\datasets\DND'  # os.path.join(ROOT, 'datasets', 'DND')
    SAVE_MAT_ROOT = os.path.join(ROOT, args.save_path, f'SubmitSrgb-DND-{args.stage}-mat')
    SAVE_IMG_ROOT = os.path.join(ROOT, args.save_path, f'SubmitSrgb-DND-{args.stage}-png')
    os.makedirs(SAVE_MAT_ROOT, exist_ok=True)
    if args.save_image:
        os.makedirs(SAVE_IMG_ROOT, exist_ok=True)

    if args.stage == 'stage1':
        ckpt = "D2S-RSG-SSD.pth"
        model = Deam(Isreal=True, scale=16)
        model = load(path=os.path.join(CKPT_ROOT, ckpt), model=model, key='',
                     delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model1 = model2 = model
    elif args.stage == 'stage2':
        ckpt = f"D2S-RSG-SSD(E)-DND.pth"
        model1 = Deam(Isreal=True, scale=16)
        model2 = Deam(Isreal=True, scale=16)
        model1 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model1, key='',
                      delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model2 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model2, key='',
                      delete_module=True, delete_str=['dn1.branches.0.'], print_=True)
    elif args.stage == 'stage3':
        ckpt = f"D2S-RSG-SSD(D)-DND.pth"
        model_distillation = Deam(Isreal=True, scale=16)
        model_distillation = load(path=os.path.join(CKPT_ROOT, ckpt), model=model_distillation,
                                  key='params', delete_module=False, print_=True)

    denoiser = TestClass()

    with h5py.File(f'{DATA_ROOT}/info.mat', 'r') as info_file:
        info = info_file['info']
        boxes_refs = info['boundingboxes']
        for i in tqdm(range(boxes_refs.shape[1]), desc='DND images'):
            name = f'{i + 1:04d}.mat'
            with h5py.File(f'{DATA_ROOT}/images_srgb/{name}', 'r') as image_file:
                image = np.asarray(image_file['InoisySRGB'], dtype=np.float32).T
            boxes = np.asarray(info[boxes_refs[0, i]]).T
            results = np.empty((len(boxes),), dtype=object)
            for k in range(len(boxes)):
                top, left, bottom, right = boxes[k].astype(int)
                crop = image[top - 1:bottom, left - 1:right, :]
                ###
                if args.stage == 'stage1' or args.stage == 'stage2':
                    # self-supervised
                    restored = denoiser.pipeline(
                        stage1_model=model1,
                        stage2_model=model2,
                        lq=crop,
                        if_refine=False if args.stage == 'stage1' else True,
                        pd_factor=2,
                        r3_num=8,
                        r3_factor=0.16
                    )
                else:
                    # self-distillation
                    restored = denoiser.distillation(
                        distillation_model=model_distillation,
                        lq=crop
                    )
                ###
                results[k] = restored
                if args.save_image:
                    Image.fromarray(np.round(restored * 255).astype(np.uint8)).save(
                        os.path.join(SAVE_IMG_ROOT, f'{i + 1:04d}_{k + 1:02d}.png'))
            sio.savemat(os.path.join(SAVE_MAT_ROOT, name), {'Idenoised': results, 'israw': False, 'eval_version': '1.0'})
    print(f'Saved DND submission files to {SAVE_MAT_ROOT}')
