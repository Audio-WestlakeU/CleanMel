import importlib
import os

import torch
import yaml
from jsonargparse import ArgumentParser
from pytorch_lightning import LightningModule
from argparse import ArgumentParser as Parser
from typing import *

import operator

from lightning_utilities.core.imports import compare_version

_TORCH_GREATER_EQUAL_2_1 = compare_version("torch", operator.ge, "2.1.0", use_base_version=True)


# this function is ported from lightning
def measure_flops(
    model: torch.nn.Module,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    total: bool = True,
) -> int:
    """Utility to compute the total number of FLOPs used by a module during training or during inference.

    It's recommended to create a meta-device model for this, because:
    1) the flops of LSTM cannot be measured if the model is not a meta-device model:

    Example::

        with torch.device("meta"):
            model = MyModel()
            x = torch.randn(2, 32)

        model_fwd = lambda: model(x)
        fwd_flops = measure_flops(model, model_fwd)

        model_loss = lambda y: y.sum()
        fwd_and_bwd_flops = measure_flops(model, model_fwd, model_loss)

    Args:
        model: The model whose FLOPs should be measured.
        forward_fn: A function that runs ``forward`` on the model and returns the result.
        loss_fn: A function that computes the loss given the ``forward_fn`` output. If provided, the loss and `backward`
            FLOPs will be included in the result.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError("`measure_flops` requires PyTorch >= 2.1.")
    from torch.utils.flop_counter import FlopCounterMode

    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        if loss_fn is None:
            forward_fn()
        else:
            loss_fn(forward_fn()).backward()
    if total:
        return flop_counter.get_total_flops()
    else:
        return flop_counter


def detailed_flops(flop_counter) -> str:
    sss = ""
    for k, v in flop_counter.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }\n"
        sss += ss
    return sss


class FakeModule(torch.nn.Module):

    def __init__(self, module: LightningModule) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module.forward(x, 0)[0]


def _get_num_params(model: torch.nn.Module):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params


def _test_FLOPs(model: LightningModule, save_dir: str, num_chns: int, fs: int, audio_time_len: int = 4, num_params: int = None):
    x = torch.randn(1, int(fs * audio_time_len), dtype=torch.float32).to("cuda")
    model = model.to("cuda")
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(model, display=False) as fcm:
        res = model(x, x, True)[0].mean()
        flops_forward_eval = fcm.get_total_flops()
    with open(os.path.join(save_dir, 'FLOPs-detailed.txt'), 'w') as f:
        for k, v in fcm.get_flop_counts().items():
            f.write(str(k) + ': { ')
            for kk, vv in v.items():
                f.write(str(kk).replace('.default', '') + ': ' + str(vv) + ', ')
            f.write(' }\n')
    params_eval = sum(param.numel() for param in model.parameters())
    flops_forward_eval_avg = flops_forward_eval/4e9
    print(f"flops_forward={flops_forward_eval_avg:.2f}G/s, params={params_eval/1e6:.2f} M")
    
    with open(os.path.join(save_dir, 'FLOPs.yaml'), 'w') as f:
        yaml.dump(
            {
                "flops_forward" if _TORCH_GREATER_EQUAL_2_1 else "macs_forward": f"{flops_forward_eval/1e9:.2f} G",
                "flops_forward_avg" if _TORCH_GREATER_EQUAL_2_1 else "macs_forward_avg": f"{flops_forward_eval_avg/1e9:.2f} G/s",
                "params": f"{params_eval/1e6:.3f} M",
                "fs": fs,
                "audio_time_len": audio_time_len,
                "num_chns": num_chns,
            }, f)
        f.close()


def import_class(class_path: str):
    try:
        iclass = importlib.import_module(class_path)
        return iclass
    except:
        imodule = importlib.import_module('.'.join(class_path.split('.')[:-1]))
        iclass = getattr(imodule, class_path.split('.')[-1])
        return iclass


def _test_FLOPs_from_config(save_dir: str, model_class_path: str, num_chns: int, fs: int, audio_time_len: int = 4, config_file: str = None):
    if config_file is None:
        config_file = os.path.join(save_dir, 'config.yaml')

    model_class = import_class(model_class_path)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, yaml.FullLoader)
    parser = ArgumentParser()
    parser.add_class_arguments(model_class)

    if 'compile' in config['model']:
        config['model']['compile'] = False  # compiled model will fail to test its flops
    try:
        if 'compile' in config['model']['arch']['init_args']:
            config['model']['arch']['init_args']['compile'] = False
    except:
        ...
    model_config = parser.instantiate_classes(config['model'])
    model = model_class(**model_config.as_dict())
    num_params = _get_num_params(model=model)
    try:
        # torcheval report error for shared modules, so config to not share
        if "full_share" in config['model']['arch']['init_args']:
            if config['model']['arch']['init_args']['full_share'] == True:
                config['model']['arch']['init_args']['full_share'] = False
                model_config = parser.instantiate_classes(config['model'])
                model = model_class(**model_config.as_dict())
            elif type(config['model']['arch']['init_args']['full_share']) == int or config['model']['arch']['init_args']['full_share'] == None:
                config['model']['arch']['init_args']['full_share'] = 9999
                model_config = parser.instantiate_classes(config['model'])
                model = model_class(**model_config.as_dict())
    except Exception as e:
        ...
    _test_FLOPs(model, save_dir=save_dir, num_chns=num_chns, fs=fs, audio_time_len=audio_time_len, num_params=num_params)


def write_FLOPs(model: LightningModule, save_dir: str, num_chns: int, fs: int = None, nfft: int = None, audio_time_len: int = 4, model_class_path: str = None):
    assert fs is not None or nfft is not None, (fs, nfft)
    if model_class_path is None:
        model_class_path = f"{str(model.__class__.__module__)}.{type(model).__name__}"
    if fs:
        cmd = f'CUDA_VISIBLE_DEVICES={model.device.index}, python -m model.utils.flops ' + f'--save_dir {save_dir} --model_class_path {model_class_path} ' + f'--num_chns {num_chns} --fs {fs} --audio_time_len {audio_time_len}'
    else:
        cmd = f'CUDA_VISIBLE_DEVICES={model.device.index}, python -m model.utils.flops ' + f'--save_dir {save_dir} --model_class_path {model_class_path} ' + f'--num_chns {num_chns} --nfft {nfft} --audio_time_len {audio_time_len}'
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = Parser()
    parser.add_argument('--save_dir', type=str, required=True, help='save FLOPs to dir')
    parser.add_argument('--model_class_path', type=str, required=True, help='the import path of your Lightning Module')
    parser.add_argument('--num_chns', type=int, required=True, help='the number of microphone channels')
    parser.add_argument('--fs', type=int, default=None, help='sampling rate')
    parser.add_argument('--nfft', type=int, default=None, help='sampling rate')
    parser.add_argument('--audio_time_len', type=float, default=4., help='seconds of test mixture waveform')
    parser.add_argument('--config_file', type=str, default=None, help='config file path')
    args = parser.parse_args()

    fs = args.fs
    if fs is None:
        if args.nfft is None:
            print('MACs test error: you should specify the fs or nfft')
            exit(-1)
        fs = {256: 8000, 512: 16000, 400: 16000}[args.nfft]
    
    _test_FLOPs_from_config(
        save_dir=args.save_dir,
        model_class_path=args.model_class_path,
        num_chns=args.num_chns,
        fs=fs,
        audio_time_len=args.audio_time_len,
        config_file=args.config_file,
    )
