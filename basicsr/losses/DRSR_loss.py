import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from basicsr.utils.registry import LOSS_REGISTRY

losses_dict = {
    'l1': nn.L1Loss(),
    'l2': nn.MSELoss()
}


@LOSS_REGISTRY.register()
class PdMaskLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1', pd_factor: int = 5, loss_weight: float = 1.0, reduction: str = 'mean', loss_mode: str = 'fast') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]
        self.pd_factor = pd_factor
        self.loss_weight = loss_weight
        self.loss_mode = loss_mode

    def forward(self, sub_denoised1: torch.tensor, sub_denoised2: torch.tensor, noisy_sub: torch.tensor,
                m_sub: torch.tensor):
        if self.loss_mode == 'old':
            return self.forward_old(sub_denoised1, sub_denoised2, noisy_sub, m_sub)
        elif self.loss_mode == 'fast':
            return self.forward_fast(sub_denoised1, sub_denoised2, noisy_sub, m_sub)
        elif self.loss_mode == 'fast_dual':
            return self.forward_fast_dual(sub_denoised1, sub_denoised2, noisy_sub, m_sub)
        else:
            NotImplementedError(f'{self.loss_mode} mode is not supported')


    def forward_old(self, sub_denoised1: torch.tensor, sub_denoised2: torch.tensor, noisy_sub: torch.tensor,
                m_sub: torch.tensor) -> torch.Tensor:
        total_loss = 0
        for j in range(self.pd_factor * self.pd_factor):
            ####################### RSGj(y.*M) - RSGj(y)   ##################
            p1 = torch.mul((1 - m_sub[j, ...]), sub_denoised1[j, ...])
            p2 = torch.mul((1 - m_sub[j, ...]), (noisy_sub[(j + 1) % (self.pd_factor * self.pd_factor), ...]))
            p3 = torch.mul(m_sub[j, ...], sub_denoised2[j, ...])
            p4 = torch.mul(m_sub[j, ...], noisy_sub[(j + 1) % (self.pd_factor * self.pd_factor), ...])
            total_loss += self.loss(p1, p2) + self.loss(p3, p4)

        return total_loss * self.loss_weight

    def forward_fast(self, sub_denoised1: torch.tensor, sub_denoised2: torch.tensor, noisy_sub: torch.tensor,
                m_sub: torch.tensor) -> torch.Tensor:
        pp, n, c, h2, w2 = noisy_sub.shape

        # 1) get noisy_sub shifted by +1 index for each j: noisy_sub[(j+1)%pp]
        #    roll(-1) moves element at pos k to position k-1, so at index j we get original j+1
        noisy_next = noisy_sub.roll(-1, dims=0)  # shape (pp, n, c, h2, w2)

        # 2) make sure mask is float 0/1
        m = m_sub.float()  # (pp, n, c, h2, w2) or (pp, n, 1, h2, w2)
        # if mask has channel dim =1, broadcast will work fine

        # 3) combine denoised according to mask
        combined = (1.0 - m) * sub_denoised1 + m * sub_denoised2  # (pp,n,c,h2,w2)
        target = noisy_next  # (pp,n,c,h2,w2)

        return self.loss(combined, target) * self.loss_weight

    def forward_fast_dual(self, sub_denoised1: torch.tensor, sub_denoised2: torch.tensor, noisy_sub: torch.tensor,
                            m_sub: torch.tensor) -> torch.Tensor:

        # 2) make sure mask is float 0/1
        m = m_sub.float()  # (pp, n, c, h2, w2) or (pp, n, 1, h2, w2)
        # if mask has channel dim =1, broadcast will work fine

        # 3) combine denoised according to mask
        combined = (1.0 - m) * sub_denoised1 + m * sub_denoised2  # (pp,n,c,h2,w2)

        dual_loss = self.loss(combined, noisy_sub.roll(-1, dims=0)) + self.loss(combined, noisy_sub.roll(1, dims=0))

        return dual_loss * self.loss_weight


@LOSS_REGISTRY.register()
class SDAPLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1', pd_factor: int = 5, loss_weight: float = 1.0, reduction: str = 'mean', loss_mode: str = 'old') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]
        self.pd_factor = pd_factor
        self.loss_weight = loss_weight
        self.loss_mode = loss_mode


    def forward(self, sub_denoised: torch.tensor, noisy_sub: torch.tensor) -> torch.Tensor:
        pp, n, c, h2, w2 = noisy_sub.shape

        # 1) get noisy_sub shifted by +1 index for each j: noisy_sub[(j+1)%pp]
        #    roll(-1) moves element at pos k to position k-1, so at index j we get original j+1
        noisy_next = noisy_sub.roll(-1, dims=0)  # shape (pp, n, c, h2, w2)

        return self.loss(sub_denoised, noisy_next) * self.loss_weight


@LOSS_REGISTRY.register()
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight: float = 1.0, reduction: str = 'mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # loss min -> SSIM max
        # 1-ssim_map.mean() | -torch.log10(ssim_map.mean()) | -ssim
        return 1-_ssim(img1, img2, window, self.window_size, channel, self.size_average) * self.loss_weight


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
