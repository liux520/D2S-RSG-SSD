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

from basicsr.models.DRSR_utils.net_func import (generate_subimages_new, subimages_reverse_new,
                                                RSG, pixel_shuffle_func, pixel_unshuffle_func, RSG_func, demask)
from basicsr.models.DRSR_utils.mask import MultiScaleMask, StableMask
from basicsr.models.DRSR_utils.R3 import R3


@MODEL_REGISTRY.register()
class DRSRBaseModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(DRSRBaseModel, self).__init__(opt)

        pass

    def _grad(self, net, requires_grad=True):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def model_ema(self, net_g, net_g_ema, decay=0.999):
        net_g = self.get_bare_model(net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_losses(self):
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('l1_opt'):
            self.cri_l1 = build_loss(train_opt['l1_opt']).to(self.device)
        else:
            self.cri_l1 = None

        if train_opt.get('l2_opt'):
            self.cri_l2 = build_loss(train_opt['l2_opt']).to(self.device)
        else:
            self.cri_l2 = None

        if train_opt.get('mask_opt'):
            self.cri_mask = build_loss(train_opt['mask_opt']).to(self.device)
        else:
            self.cri_mask = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

        if train_opt.get('psnr_opt'):
            self.cri_psnr = build_loss(train_opt['psnr_opt']).to(self.device)
        else:
            self.cri_psnr = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None



        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        ### TODO: new codes
        noisy_sub, noisy_sub1, noisy_sub2, noisy_sub1_denoised, noisy_sub2_denoised, mask_sub, location_mask = (
            RSG(self.lq, self.pd_factor_train, self.mask))
        for j in range(self.pd_factor_train * self.pd_factor_train):  # pd_factor * pd_factor subimages
            noisy_sub1_denoised[j, ...] = self.net_g(noisy_sub1[j, ...])
            noisy_sub2_denoised[j, ...] = self.net_g(noisy_sub2[j, ...])

        # n, b, c, h, w = noisy_sub1_denoised.shape
        self.output = (
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

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(noisy_sub1_denoised, noisy_sub2_denoised, noisy_sub, mask_sub)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.lq)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_edge:
            l_edge = self.cri_edge(self.output, self.lq)
            l_total += l_edge
            loss_dict['l_edge'] = l_edge
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.lq)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
        if self.cri_psnr:
            l_psnr = self.cri_psnr(self.output, self.lq)
            l_total += l_psnr
            loss_dict['l_psnr'] = l_psnr

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

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
            self.test()

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


