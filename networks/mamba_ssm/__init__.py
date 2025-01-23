'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-10-08 15:26:43
FilePath: /InASR/networks/mamba_ssm/__init__.py
'''
__version__ = "1.2.0.post1"

from networks.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from networks.mamba_ssm.modules.mamba_simple import Mamba

