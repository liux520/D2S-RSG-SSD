import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.DRSR_utils.mask import MultiScaleMask, StableMask


class MaskEnhance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask = MultiScaleMask(6)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor, net: nn.Module) -> torch.Tensor:
        masked_img, masks = self.mask(denoised)
        dn_img = torch.zeros_like(x)
        for i in range(len(masked_img)):
            masked_img[i][~masks[i]] = x[~masks[i]]
            out = net(masked_img[i])
            dn_img += out
        return dn_img / len(masked_img)


class R3(nn.Module):
    def __init__(self, r3_factor: float = -1, r3_num: int = 8):
        super().__init__()
        self.r3_factor = r3_factor
        self.r3_num = r3_num
        # if r3_factor <= 0:
        #     self.enhance = MaskEnhance()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor, net: nn.Module) -> torch.Tensor:
        if self.r3_factor > 0:
            return r3(x, denoised, net, self.r3_factor, self.r3_num)
        else:
            # return self.enhance(x, denoised, net)
            return denoised


def r3(x: torch.Tensor, denoised: torch.Tensor, net: nn.Module, r3_factor: float = 0.16, r3_num: int = 8, p: int = 0) -> torch.Tensor:
    """random replacement Refinement with ratio r3_factor

    Note:
        This module is only used in eval, not in train. val will take r3_num times longer than train.
    Args:
        x(torch.Tensor): input tensor.BCHW,B=1
        net(nn.Module): model to eval
        r3_factor (float, optional): the ratio of radnom replace. Defaults to 0.16.
        r3_num (int, optional): the number of r3 times. Defaults to 8.
    Output: BCHW
    """

    # im_denoise = torch.cat([self.dn2(im_denoise[:, 0, :, :].unsqueeze(1)),
    #             self.dn2(im_denoise[:, 1, :, :].unsqueeze(1)),
    #             self.dn2(im_denoise[:, 2, :, :].unsqueeze(1))], dim=1)

    b, c, h, w = x.shape
    temp_input = denoised.expand(r3_num, -1, -1, -1)
    x = x.expand(r3_num, -1, -1, -1)
    indices = torch.zeros(r3_num, c, h, w, dtype=torch.bool, device=x.device)
    for t in range(r3_num):
        indices[t] = (torch.rand(1, h, w) < r3_factor).repeat(3, 1, 1)
    temp_input = temp_input.clone()
    temp_input[indices] = x[indices]
    temp_input = F.pad(temp_input, (p, p, p, p), mode='reflect')
    with torch.no_grad():
        if p == 0:
            denoised = net(temp_input)
            # denoised = torch.cat([net(temp_input[:, 0, :, :].unsqueeze(1)),net(temp_input[:, 1, :, :].unsqueeze(1)),net(temp_input[:, 2, :, :].unsqueeze(1))],dim=1)
        else:
            denoised = net(temp_input)[:, :, p:-p, p:-p]
            # denoised = torch.cat([net(temp_input[:, 0, :, :].unsqueeze(1))[:, :, p:-p, p:-p],net(temp_input[:, 1, :, :].unsqueeze(1))[:, :, p:-p, p:-p],net(temp_input[:, 2, :, :].unsqueeze(1))][:, :, p:-p, p:-p],dim=1)
    return torch.mean(denoised, dim=0).unsqueeze(0)
