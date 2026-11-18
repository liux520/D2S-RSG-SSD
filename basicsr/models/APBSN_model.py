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
from basicsr.models.DRSR_utils import util


@MODEL_REGISTRY.register()
class MMBSNModel(DRSRBaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(MMBSNModel, self).__init__(opt)

        # define stage1 network
        self.net_g1 = build_network(opt['network_g1'])
        self.net_g1 = self.model_to_device(self.net_g1)
        self._grad(self.net_g1, opt['network_g1']['requires_grad'])

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g1', 'params')
            self.load_network(self.net_g1, load_path, self.opt['path'].get('strict_load_g1', True), param_key)

        if self.is_train:
            self.init_training_settings()

        ###
        self.pd_factor_train = opt['train']['pd_factor_train']
        self.pd_factor_test = opt['train']['pd_factor_test']
        # self.mask_mode = opt['train']['mask_mode']

        self.r3 = R3(
            r3_factor=opt['train']['r3_factor'],  # the ratio of radnom replace. Defaults to 0.16.
            r3_num=opt['train']['r3_num']  # the number of r3 times. Defaults to 8.
        )
        self.r3_num = opt['train']['r3_num']
        self.r3_factor = opt['train']['r3_factor']

        ###

    def init_training_settings(self):
        self.net_g1.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g1_ema = build_network(self.opt['network_g1']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g1', None)
            if load_path is not None:
                self.load_network(self.net_g1_ema, load_path, self.opt['path'].get('strict_load_g1', True), 'params_ema')
            else:
                self.model_ema(self.net_g1, self.net_g1_ema, 0)  # copy net_g weight
            self.net_g1_ema.eval()

        # set up optimizers and schedulers
        self.setup_losses()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params1 = []
        for k, v in self.net_g1.named_parameters():
            if v.requires_grad:
                optim_params1.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type,
            optim_params1,
            **train_opt['optim_g']
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        target = self.lq.clone()

        x = util.pd_down(self.lq, self.pd_factor_train)
        out = self.net_g1(x)
        dn_img = util.pd_up(out, self.pd_factor_train)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_l1:
            l_mask = self.cri_l1(target, dn_img)
            l_total += l_mask
            loss_dict['l_mask'] = l_mask

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(self.net_g1, self.net_g1_ema, decay=self.ema_decay)

    @torch.inference_mode()
    def test(self, mode='standard', _apply_R3=False):
        self.test_stage1(mode, _apply_R3)

    @torch.inference_mode()
    def test_stage1(self, mode='standard', _apply_R3=False):
        assert mode in ('standard', 'ema'), f'{mode} is not supported, only standard/ema mode is supported.'
        if mode == 'standard':
            self.net_g1.eval()
            target = self.lq.clone()
            x = util.pd_down(self.lq, self.pd_factor_test)
            out = self.net_g1(x)
            self.output = util.pd_up(out, self.pd_factor_test)

            if _apply_R3:
                self.output = self.r3(target, self.output, self.net_g1)

            self.net_g1.train()

        elif mode == 'ema':
            self.net_g1_ema.eval()
            target = self.lq.clone()
            x = util.pd_down(self.lq, self.pd_factor_test)
            out = self.net_g1_ema(x)
            self.output = util.pd_up(out, self.pd_factor_test)

            if _apply_R3:
                self.output = self.r3(target, self.output, self.net_g1_ema)

        else:
            NotImplementedError(f'{mode} is not supported, only standard/ema mode is supported.')

    @torch.inference_mode()
    def apply_R3(self, net, noisy, denoised, p=0):
        # [1, c, h, w]
        b, c, h, w = noisy.shape
        # denoised: [r3_num, c, h, w]
        temp_input = denoised.expand(self.r3_num, -1, -1, -1)
        # noisy:    [r3_num, c, h, w]
        x = noisy.expand(self.r3_num, -1, -1, -1)
        indices = torch.zeros(self.r3_num, c, h, w, dtype=torch.bool, device=x.device)
        for t in range(self.r3_num):
            indices[t] = (torch.rand(1, h, w) < self.r3_factor).repeat(3, 1, 1)
        temp_input = temp_input.clone()
        temp_input[indices] = x[indices]
        if p != 0:
            temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
        with torch.no_grad():
            if p == 0:
                denoised = net(temp_input)
            else:
                denoised = net(temp_input)[:, :, p:-p, p:-p]
        return torch.mean(denoised, dim=0).unsqueeze(0)

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
            self.test(mode=self.opt['val']['mode'], _apply_R3=self.opt['val']['apply_R3'])

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
        if hasattr(self, 'net_g1_ema'):
            self.save_network([self.net_g1, self.net_g1_ema], 'net_g1', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g1, 'net_g1', current_iter)
        self.save_training_state(epoch, current_iter)
