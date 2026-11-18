import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
# from ptflops import get_model_complexity_info
import torch
from torchvision.models import resnet50
from thop import clever_format, profile


@torch.inference_mode()
def params_flops(model, input_tensor, use_cuda=True, model_name=''):
    if use_cuda:
        model.cuda()
        dummy_input = input_tensor.cuda()
    macs, params = profile(model, inputs=(dummy_input,))

    # Format the numbers into a readable format (e.g., 4.14 GMac, 25.56 MParams)
    macs_readable, params_readable = clever_format([macs, params], "%.3f")

    print(f"{model_name}: Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
    # Expected output: Formatted MACs: 4.140G, Formatted Parameters: 25.557M


    # with torch.cuda.device(0):
    #     net = models.densenet161()
    #     macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, backend='pytorch'
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    #
    #     macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, backend='aten'
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def benchmark_latency(model, input_tensor, runs=100, use_cuda=True, model_name=''):
    model.eval()
    if use_cuda:
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Warm up
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    total = time.time() - start

    print(f"{model_name} Average latency: {total / runs * 1000:.3f} ms")


def benchmark_latency_R3(model, input_tensor, runs=100, use_cuda=True, model_name='', _apply_r3=True):
    model.eval()
    if use_cuda:
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Warm up
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in tqdm(range(runs)):
            x = pixel_shuffle_down_sampling(input_tensor, 2)
            _ = model(x)
            x = pixel_shuffle_up_sampling(x, 2)
            if _apply_r3:
                x = r3(input_tensor, x, model)


    torch.cuda.synchronize()
    total = time.time() - start

    print(f"{model_name} Average latency: {total / runs * 1000:.3f} ms")


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,
                                                                                                           w + 2 * f * pad,
                                                                                                           h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c,
                                                                                                                 w + 2 * f * pad,
                                                                                                                 h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0):
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
                                                                                              h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def r3(x, denoised, net, r3_factor: float = 0.16, r3_num: int = 8, p: int = 0) -> torch.Tensor:
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
        else:
            denoised = net(temp_input)[:, :, p:-p, p:-p]
    return torch.mean(denoised, dim=0).unsqueeze(0)


if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 512, 512)

    # DeamNet
    from basicsr.archs.deamnet_arch import Deam
    model = Deam()
    # benchmark_latency(model, input_tensor, model_name='Deam')
    params_flops(model, input_tensor, model_name='Deam')

    # MMBSN
    from basicsr.archs.MMBSN_arch import MMBSN
    model = MMBSN()
    # benchmark_latency_R3(model, input_tensor, model_name='MMBSN', _apply_r3=False)
    params_flops(model, input_tensor, model_name='MMBSN')

    # Restormer
    from basicsr.archs.restormer_arch import Restormer
    model = Restormer()
    # benchmark_latency(model, input_tensor, model_name='Restormer')
    params_flops(model, input_tensor, model_name='Restormer')