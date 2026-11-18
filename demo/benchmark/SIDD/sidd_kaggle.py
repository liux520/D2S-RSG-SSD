import wget
import argparse
import os,  torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import scipy.io
import numpy as np
import pandas as pd
import base64
from demo.test_utils import load, pixel_shuffle_func, pixel_unshuffle_func, str2bool
from basicsr.archs.deamnet_arch import Deam
from basicsr.models.DRSR_utils.mask import StableMask, MultiScaleMask
import torch.nn.functional as F
"""
https://github.com/AbdoKamel/sidd_kaggle_submit/blob/main/prepare_submission_srgb.py
https://abdokamel.github.io/sidd/

https://www.kaggle.com/c/sidd-benchmark-srgb-psnr
https://www.kaggle.com/c/sidd-benchmark-srgb-ssim
"""


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


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

        lq_tensor = uint2tensor(lq).to(self.device)

        output = self.stage_1(stage1_model, lq_tensor, pd_factor)
        if if_refine:
            output = self.stage_2(stage2_model, lq_tensor, output, pd_factor, r3_num, r3_factor)

        return tensor2uint8(output)

    @torch.inference_mode()
    def distillation(self, distillation_model, lq):

        distillation_model.to(self.device).eval()
        lq_tensor = uint2tensor(lq).to(self.device)
        output = distillation_model(lq_tensor)

        return tensor2uint8(output)


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], default='stage3')
    parser.add_argument('--save_path', type=str, default=r'results/submit/SIDD')
    parser.add_argument('--save_image', type=str2bool, default=False)
    args = parser.parse_args()

    ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
    CKPT_ROOT = os.path.join(ROOT, 'ckpts')
    DATA_ROOT = r'G:\PhD\Paper\LAB\WEIGUANG\TPAMI\Compare\datasets\SIDDBenchamrk'  # os.path.join(ROOT, 'datasets', 'SIDDBenchamrk')
    SAVE_CSV_ROOT = os.path.join(ROOT, args.save_path)
    SAVE_IMG_ROOT = os.path.join(ROOT, args.save_path, f'SubmitSrgb-SIDD-{args.stage}')
    os.makedirs(SAVE_CSV_ROOT, exist_ok=True)
    if args.save_image:
        os.makedirs(SAVE_IMG_ROOT, exist_ok=True)
    output_file = os.path.join(SAVE_CSV_ROOT, f'SubmitSrgb-SIDD-{args.stage}.csv')

    input_file = os.path.join(DATA_ROOT, 'BenchmarkNoisyBlocksSrgb.mat')
    inputs = scipy.io.loadmat(input_file)['BenchmarkNoisyBlocksSrgb']
    print(f'inputs.shape = {inputs.shape} | inputs.dtype = {inputs.dtype}')

    if args.stage == 'stage1':
        ckpt = "D2S-RSG-SSD.pth"
        model = Deam(Isreal=True, scale=16)
        model = load(path=os.path.join(CKPT_ROOT, ckpt), model=model, key='',
                     delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model1 = model2 = model
    elif args.stage == 'stage2':
        ckpt = f"D2S-RSG-SSD(E)-SIDD.pth"
        model1 = Deam(Isreal=True, scale=16)
        model2 = Deam(Isreal=True, scale=16)
        model1 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model1, key='',
                      delete_module=True, delete_str=['dn.branches.0.'], print_=True)
        model2 = load(path=os.path.join(CKPT_ROOT, ckpt), model=model2, key='',
                      delete_module=True, delete_str=['dn1.branches.0.'], print_=True)
    elif args.stage == 'stage3':
        ckpt = f"D2S-RSG-SSD(D)-SIDD.pth"
        model_distillation = Deam(Isreal=True, scale=16)
        model_distillation = load(path=os.path.join(CKPT_ROOT, ckpt), model=model_distillation,
                                  key='params', delete_module=False, print_=True)

    denoiser = TestClass()

    output_blocks_base64string = []
    for i in tqdm(range(inputs.shape[0])):
        for j in range(inputs.shape[1]):
            in_block = inputs[i, j, :, :, :]
            ###
            if args.stage == 'stage1' or args.stage == 'stage2':
                # self-supervised
                out_block = denoiser.pipeline(
                    stage1_model=model1,
                    stage2_model=model2,
                    lq=in_block,
                    if_refine=False if args.stage == 'stage1' else True,
                    pd_factor=2,
                    r3_num=8,
                    r3_factor=0.16
                )
            else:
                # self-distillation
                out_block = denoiser.distillation(
                    distillation_model=model_distillation,
                    lq=in_block
                )

            if args.save_image:
                Image.fromarray(out_block).save(os.path.join(SAVE_IMG_ROOT, f'{i:04d}_{j:04d}.png'))

            ###
            # assert in_block.shape == out_block.shape
            # assert in_block.dtype == out_block.dtype
            out_block_base64string = array_to_base64string(out_block)
            output_blocks_base64string.append(out_block_base64string)

    print(f'Saving outputs to {output_file}')
    output_df = pd.DataFrame()
    n_blocks = len(output_blocks_base64string)
    print(f'Number of blocks = {n_blocks}')
    output_df['ID'] = np.arange(n_blocks)
    output_df['BLOCK'] = output_blocks_base64string
    output_df.to_csv(output_file, index=False)

    # kaggle.com/competitions/sidd-benchmark-srgb-psnr
    # kaggle competitions submit -c sidd-benchmark-srgb-psnr -f submission.csv -m "Message"
    print('TODO: Submit the output file SubmitSrgb.csv at ')
    print('kaggle.com/competitions/sidd-benchmark-srgb-psnr')
    print('Done.')
