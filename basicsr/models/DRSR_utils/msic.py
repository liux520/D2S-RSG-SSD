import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def set_require_grad(net: torch.nn.Module | list, requires_grad: bool):
    if isinstance(net, list):
        for n in net:
            set_require_grad(n, requires_grad)
        return
    for param in net.parameters():
        param.requires_grad = requires_grad


def clip_grad_norm(nets, norm: float):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=norm)


def initialize_weights(net_l, scale: float = 1):
    """
    初始化网络参数
    :param net_l: 要初始化的网络
    :param scale: 倍数
    :return: none
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            # kaiming 正态分布
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)