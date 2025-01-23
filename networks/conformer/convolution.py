'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-05-18 16:46:10
FilePath: /espnet/espnet/nets/pytorch_backend/conformer/convolution.py
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn


##causal
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self,x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        if self.lorder > 0:
            x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim) 

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)



# class ConvolutionModule(nn.Module):
#     """ConvolutionModule in Conformer model.

#     Args:
#         channels (int): The number of channels of conv layers.
#         kernel_size (int): Kernerl size of conv layers.

#     """

#     def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
#         """Construct an ConvolutionModule object."""
#         super(ConvolutionModule, self).__init__()
#         # kernerl_size should be a odd number for 'SAME' padding
#         assert (kernel_size - 1) % 2 == 0

#         self.pointwise_conv1 = nn.Conv1d(
#             channels,
#             2 * channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#         self.depthwise_conv = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size,
#             stride=1,
#             padding=(kernel_size - 1) // 2,
#             groups=channels,
#             bias=bias,
#         )
#         self.norm = nn.BatchNorm1d(channels)
#         self.pointwise_conv2 = nn.Conv1d(
#             channels,
#             channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=bias,
#         )
#         self.activation = activation

#     def forward(self, x):
#         """Compute convolution module.

#         Args:
#             x (torch.Tensor): Input tensor (#batch, time, channels).

#         Returns:
#             torch.Tensor: Output tensor (#batch, time, channels).

#         """
#         # exchange the temporal dimension and the feature dimension
#         x = x.transpose(1, 2)

#         # GLU mechanism
#         x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
#         x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

#         # 1D Depthwise Conv
#         x = self.depthwise_conv(x)
#         x = self.activation(self.norm(x))

#         x = self.pointwise_conv2(x)

#         return x.transpose(1, 2)
