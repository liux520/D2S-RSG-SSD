import itertools
import torch.nn.functional as F
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.models.DRSR_base_model import DRSRBaseModel
from basicsr.models.DRSR_utils.net_func import (generate_subimages_new, subimages_reverse_new,
                                                RSG, pixel_shuffle_func, pixel_unshuffle_func, RSG_func, demask)
from basicsr.models.DRSR_utils.mask import MultiScaleMask, StableMask
from basicsr.models.DRSR_utils.R3 import R3
from basicsr.models.DRSR_utils.cpp_fast import forward_fast_last


@MODEL_REGISTRY.register()
class DRSRDualModel(DRSRBaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DRSRDualModel, self).__init__(opt)

        # define stage1 network
        self.net_g1 = build_network(opt['network_g1'])
        self.net_g1 = self.model_to_device(self.net_g1)
        self._grad(self.net_g1, opt['network_g1']['requires_grad'])

        # define stage1 network
        self.net_g2 = build_network(opt['network_g2'])
        self.net_g2 = self.model_to_device(self.net_g2)
        self._grad(self.net_g2, opt['network_g2']['requires_grad'])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g1', 'params')
            self.load_network(self.net_g1, load_path, self.opt['path'].get('strict_load_g1', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g2', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g2', 'params')
            self.load_network(self.net_g2, load_path, self.opt['path'].get('strict_load_g2', True), param_key)

        if self.is_train:
            self.init_training_settings()

        ###
        self.pd_factor_train = opt['train']['pd_factor_train']
        self.pd_factor_test = opt['train']['pd_factor_test']
        # self.mask_mode = opt['train']['mask_mode']

        self.multiscale_mask = MultiScaleMask()
        self.stable_mask = StableMask()

        self.r3 = R3(
            r3_factor=opt['train']['r3_factor'],  # the ratio of radnom replace. Defaults to 0.16.
            r3_num=opt['train']['r3_num']  # the number of r3 times. Defaults to 8.
        )
        self.r3_num = opt['train']['r3_num']
        self.r3_factor = opt['train']['r3_factor']
        ###

    def init_training_settings(self):
        self.net_g1.train()
        self.net_g2.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel

            if (self.opt['network_g1']['requires_grad']):
                self.net_g1_ema = build_network(self.opt['network_g1']).to(self.device)
                load_path = self.opt['path'].get('pretrain_network_g1', None)
                if load_path is not None:
                    self.load_network(self.net_g1_ema, load_path, self.opt['path'].get('strict_load_g1', True),
                                      'params_ema')
                else:
                    self.model_ema(self.net_g1, self.net_g1_ema, 0)  # copy net_g weight
                self.net_g1_ema.eval()

            if (self.opt['network_g2']['requires_grad']):
                self.net_g2_ema = build_network(self.opt['network_g2']).to(self.device)
                load_path = self.opt['path'].get('pretrain_network_g2', None)
                if load_path is not None:
                    self.load_network(self.net_g2_ema, load_path, self.opt['path'].get('strict_load_g2', True),
                                      'params_ema')
                else:
                    self.model_ema(self.net_g2, self.net_g2_ema, 0)  # copy net_g weight
                self.net_g2_ema.eval()

        # set up optimizers and schedulers
        self.setup_losses()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params1, optim_params2 = [], []
        for k, v in self.net_g1.named_parameters():
            if v.requires_grad:
                optim_params1.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Net1: Params {k} will not be optimized.')

        for k, v in self.net_g2.named_parameters():
            if v.requires_grad:
                optim_params2.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Net2: Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type,
            # optim_params,
            itertools.chain(optim_params1, optim_params2),
            **train_opt['optim_g']
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lq_light' in data:
            self.lq_light = data['lq_light'].to(self.device)

    """
    1. train stage1 using SIDD, train RSG + test StableMask
    2. train stage2 using SIDD light noise, train RSG + test StableMask
    3. combine stage1 + stage2 using SIDD, train RSG + test StableMask
    """

    def train_single_stage_slow(self, lq, net, current_iter):
        ### TODO: new codes
        noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, location_mask = (
            RSG(lq, self.pd_factor_train, self.stable_mask)
        )
        for j in range(self.pd_factor_train * self.pd_factor_train):  # pd_factor * pd_factor subimages
            noisy_sub1_denoised[j, ...] = net(noisy_sub1[j, ...])
            noisy_sub2_denoised[j, ...] = net(noisy_sub2[j, ...])

        # n, b, c, h, w = noisy_sub1_denoised.shape
        output = (
                subimages_reverse_new(
                    subimages=torch.mul(1 - mask_sub, noisy_sub1_denoised),
                    masks=location_mask,
                    p=self.pd_factor_train
                )
                +
                subimages_reverse_new(
                    subimages=torch.mul(mask_sub, noisy_sub2_denoised),
                    masks=location_mask,
                    p=self.pd_factor_train
                )
        )
        return output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub
        ###
        ### old codes
        # noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, noisy_sub, mask_sub, location_mask = (
        #     RSG_func(self.lq, self.pd_factor_train))
        # for j in range(self.pd_factor_train * self.pd_factor_train):  # model
        #     noisy_sub1_denoised[j, ...] = self.net_g(noisy_sub1[j, ...])
        #     noisy_sub2_denoised[j, ...] = self.net_g(noisy_sub2[j, ...])
        # out1 = demask(torch.mul(1 - mask_sub, noisy_sub1_denoised), location_mask, self.pd_factor_train)
        # out2 = demask(torch.mul(mask_sub, noisy_sub2_denoised), location_mask, self.pd_factor_train)
        # self.output = out1 + out2
        ###

    def train_single_stage_fast(self, lq, net, current_iter):
        (output, part1, part2, noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised,
         mask_sub, perm_idx) = forward_fast_last(lq, net, self.pd_factor_train)
        return output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        if (self.opt['network_g1']['requires_grad']) and (not self.opt['network_g2']['requires_grad']):
            # 1. train stage1 and fix stage2
            output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub = (
                self.train_single_stage_fast(self.lq, self.net_g1, current_iter)
            )
            # 2. feed stage1 output into stage2
            self.test_stage2(mode='standard', input=output)
            output = self.output_s2
            # self.test_stage2_apply_R3(mode='standard', noisy_input=self.lq, output_s1=output)
            # output = self.output_s2_r3
        if (not self.opt['network_g1']['requires_grad']) and (self.opt['network_g2']['requires_grad']):
            if (current_iter // self.opt['train']['switch_interval']) % 2 == 0:
                # fix stage1 and train stage2
                # 1. fix stage1 for inference (self.output_s1 will be generated.)
                self.test_stage1(mode='standard', input=self.lq)
                # 2. feed stage1 output into stage2
                output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub = (
                    self.train_single_stage_fast(self.output_s1, self.net_g2, current_iter)
                )
            else:
                output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub = (
                    self.train_single_stage_fast(self.lq_light, self.net_g2, current_iter)
                )
        if (self.opt['network_g1']['requires_grad']) and (self.opt['network_g2']['requires_grad']):
            # train stage1 and train stage2
            output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub = (
                self.train_single_stage_fast(self.lq, self.net_g1, current_iter)
            )
            output, noisy_sub, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub = (
                self.train_single_stage_fast(output, self.net_g2, current_iter)
            )

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_mask:
            l_mask = self.cri_mask(noisy_sub1_denoised, noisy_sub2_denoised, noisy_sub, mask_sub)
            l_total += l_mask
            loss_dict['l_mask'] = l_mask
        if self.cri_l1:
            l_l1 = self.cri_l1(output, self.gt)
            l_total += l_l1
            loss_dict['l_l1'] = l_l1
        if self.cri_l2:
            l_l2 = self.cri_l2(output, self.gt)
            l_total += l_l2
            loss_dict['l_l2'] = l_l2
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.cri_ssim:
            l_ssim = self.cri_ssim(output, self.lq)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim
        if self.cri_tv:
            l_tv = self.cri_tv(output)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_edge:
            l_edge = self.cri_edge(output, self.lq)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge
        if self.cri_fft:
            l_fft = self.cri_fft(output, self.lq)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
        if self.cri_psnr:
            l_psnr = self.cri_psnr(output, self.lq)
            l_total += l_psnr
            loss_dict['l_psnr'] = l_psnr

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            if (self.opt['network_g1']['requires_grad']):
                self.model_ema(self.net_g1, self.net_g1_ema, decay=self.ema_decay)
            if (self.opt['network_g2']['requires_grad']):
                self.model_ema(self.net_g2, self.net_g2_ema, decay=self.ema_decay)

    @torch.inference_mode()
    def test(self, mode, apply_R3=False, stage1=True, stage2=True):
        if stage1:
            self.test_stage1(mode, input=self.lq)
            self.output = self.output_s1
        if stage2:
            if apply_R3:
                self.test_stage2_apply_R3(mode, noisy_input=self.lq, output_s1=self.output_s1)
                self.output = self.output_s2_r3
            else:
                self.test_stage2(mode, input=self.output_s1)
                self.output = self.output_s2

    @torch.inference_mode()
    def test_stage1(self, mode, input):
        assert mode in ('standard', 'ema'), f'{mode} is not supported, only standard/ema mode is supported.'
        if mode == 'standard':
            self.net_g1.eval()
            pd = pixel_unshuffle_func(input, self.pd_factor_test)  # y.*M-->PD
            pd_mask, mask = self.stable_mask(pd)
            pd1 = self.net_g1(pd_mask[0])  # PDdown-->-y.M->model
            pd2 = self.net_g1(pd_mask[1])
            test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
            test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
            test_out1 = pixel_shuffle_func(test_out1, 2)  # y.*M-->PDdown-->model-->PDup
            test_out2 = pixel_shuffle_func(test_out2, 2)
            self.output_s1 = test_out1 + test_out2
            self.net_g1.train()
        elif mode == 'ema':
            if (self.opt['network_g1']['requires_grad']):
                self.net_g1_ema.eval()
                pd = pixel_unshuffle_func(input, self.pd_factor_test)  # y.*M-->PD
                pd_mask, mask = self.stable_mask(pd)
                pd1 = self.net_g1_ema(pd_mask[0])  # PDdown-->-y.M->model
                pd2 = self.net_g1_ema(pd_mask[1])
                test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
                test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
                test_out1 = pixel_shuffle_func(test_out1, 2)  # y.*M-->PDdown-->model-->PDup
                test_out2 = pixel_shuffle_func(test_out2, 2)
                self.output_s1 = test_out1 + test_out2
            else:
                NotImplementedError(f'network_g1 has no gradient, therefore net_g1_ema will not be inited.')
        else:
            NotImplementedError(f'{mode} is not supported, only standard/ema mode is supported.')

    @torch.inference_mode()
    def test_stage2(self, mode, input):
        assert mode in ('standard', 'ema'), f'{mode} is not supported, only standard/ema mode is supported.'
        if mode == 'standard':
            self.net_g2.eval()
            pd = pixel_unshuffle_func(input, self.pd_factor_test)  # y.*M-->PD
            pd_mask, mask = self.stable_mask(pd)
            pd1 = self.net_g2(pd_mask[0])  # PDdown-->-y.M->model
            pd2 = self.net_g2(pd_mask[1])
            test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
            test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
            test_out1 = pixel_shuffle_func(test_out1, 2)  # y.*M-->PDdown-->model-->PDup
            test_out2 = pixel_shuffle_func(test_out2, 2)
            self.output_s2 = test_out1 + test_out2
            self.net_g2.train()
        elif mode == 'ema':
            if (self.opt['network_g2']['requires_grad']):
                self.net_g2_ema.eval()
                pd = pixel_unshuffle_func(input, self.pd_factor_test)  # y.*M-->PD
                pd_mask, mask = self.stable_mask(pd)
                pd1 = self.net_g2_ema(pd_mask[0])  # PDdown-->-y.M->model
                pd2 = self.net_g2_ema(pd_mask[1])
                test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
                test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
                test_out1 = pixel_shuffle_func(test_out1, 2)  # y.*M-->PDdown-->model-->PDup
                test_out2 = pixel_shuffle_func(test_out2, 2)
                self.output_s2 = test_out1 + test_out2
            else:
                NotImplementedError(f'network_g2 has no gradient, therefore net_g2_ema will not be inited.')
        else:
            NotImplementedError(f'{mode} is not supported, only standard/ema mode is supported.')

    @torch.inference_mode()
    def test_stage2_apply_R3(self, mode, noisy_input, output_s1):
        assert mode in ('standard', 'ema'), f'{mode} is not supported, only standard/ema mode is supported.'
        if mode == 'standard':
            self.net_g2.eval()
            p = 0
            b, c, h, w = output_s1.shape
            temp_input = output_s1.expand(self.r3_num, -1, -1, -1)
            lq_tensor = noisy_input.expand(self.r3_num, -1, -1, -1)
            indices = torch.zeros(self.r3_num, c, h, w, dtype=torch.bool, device=lq_tensor.device)
            for t in range(self.r3_num):
                indices[t] = (torch.rand(1, h, w) < self.r3_factor).repeat(3, 1, 1)
                temp_input = temp_input.clone()
                temp_input[indices] = lq_tensor[indices]
                temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
            if p == 0:
                pd = pixel_unshuffle_func(temp_input, 1)  # y.*M-->PD
                pd_mask, mask = self.multiscale_mask(pd)
                pd1 = self.net_g2(pd_mask[0])  # PDdown-->-y.M->model
                pd2 = self.net_g2(pd_mask[1])
                test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
                test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
                test_out1 = pixel_shuffle_func(test_out1, 1)  # y.*M-->PDdown-->model-->PDup
                test_out2 = pixel_shuffle_func(test_out2, 1)
                output_2 = test_out1 + test_out2
            self.output_s2_r3 = torch.mean(output_2, dim=0).unsqueeze(0)
            self.net_g2.train()
        elif mode == 'ema':
            if (self.opt['network_g2']['requires_grad']):
                self.net_g2_ema.eval()
                p = 0
                b, c, h, w = output_s1.shape
                temp_input = output_s1.expand(self.r3_num, -1, -1, -1)
                lq_tensor = noisy_input.expand(self.r3_num, -1, -1, -1)
                indices = torch.zeros(self.r3_num, c, h, w, dtype=torch.bool, device=lq_tensor.device)
                for t in range(self.r3_num):
                    indices[t] = (torch.rand(1, h, w) < self.r3_factor).repeat(3, 1, 1)
                    temp_input = temp_input.clone()
                    temp_input[indices] = lq_tensor[indices]
                    temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
                if p == 0:
                    pd = pixel_unshuffle_func(temp_input, 1)  # y.*M-->PD
                    pd_mask, mask = self.multiscale_mask(pd)
                    pd1 = self.net_g2_ema(pd_mask[0])  # PDdown-->-y.M->model
                    pd2 = self.net_g2_ema(pd_mask[1])
                    test_out2 = torch.mul(mask[0], pd2)  # mask[0]--->M  mask[1]-->1-M
                    test_out1 = torch.mul(mask[1], pd1)  # y.*M-->PDdown-->model-->PDup-->PDup.*(1-M)
                    test_out1 = pixel_shuffle_func(test_out1, 1)  # y.*M-->PDdown-->model-->PDup
                    test_out2 = pixel_shuffle_func(test_out2, 1)
                    output_2 = test_out1 + test_out2
                self.output_s2_r3 = torch.mean(output_2, dim=0).unsqueeze(0)
            else:
                NotImplementedError(f'network_g2 has no gradient, therefore net_g2_ema will not be inited.')
        else:
            NotImplementedError(f'{mode} is not supported, only standard/ema mode is supported.')

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test(
                mode=self.opt['val']['mode'], _apply_R3=self.opt['val']['apply_R3'],
                stage1=self.opt['val']['stage1'], stage2=self.opt['val']['stage2']
            )

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if (self.opt['network_g1']['requires_grad']):
            if hasattr(self, 'net_g1_ema'):
                self.save_network([self.net_g1, self.net_g1_ema], 'net_g1', current_iter,
                                  param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g1, 'net_g1', current_iter)

        if (self.opt['network_g2']['requires_grad']):
            if hasattr(self, 'net_g2_ema'):
                self.save_network([self.net_g2, self.net_g2_ema], 'net_g2', current_iter,
                                  param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g2, 'net_g2', current_iter)

        self.save_training_state(epoch, current_iter)
