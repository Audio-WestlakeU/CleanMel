# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
# from typeguard import typechecked

from networks.ctc import CTC
from utils.nets_utils import make_pad_mask
from networks.transformer.embedding import PositionalEncoding
from networks.transformer.layer_norm import LayerNorm
from utils.nets_utils import get_activation
from networks.conformer.convolution import ConvolutionModule

from networks.transformer.repeat import repeat
from networks.transformer.subsampling import (
    Conv2dSubsampling,
    TooShortUttError,
    check_short_utt,
)

import logging
import math
from functools import partial

import torch
import torch.nn as nn

from  networks.mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from networks.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass, field

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-12,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)




class CausalConv2dSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(CausalConv2dSubsampling, self).__init__()
        self.padding = (kernel_size[0] - 1, 0) 

        self.subsample1 = nn.Sequential(
            nn.Conv2d(
                1, 
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=2,
                bias=bias,
            ),
            nn.ReLU(),
        )

        self.subsample2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=2,
                bias=bias,
            ),
            nn.ReLU(),
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(out_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels),
            torch.nn.Dropout(0.1),
            # PositionalEncoding(out_channels, 0.1),
        )
        # self.out = torch.nn.Linear(out_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels)

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.subsample1(x)
        if self.padding[0] != 0:
            x = x[:, :, :-self.padding[0], :]
        x = self.subsample2(x)
        if self.padding[0] != 0:
            x = x[:, :, :-self.padding[0], :]
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsamplingM(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsamplingM, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
    

class MambaEncoder(nn.Module):
    """Transformer encoder module.

    Args:

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = False,
        padding_idx: int = -1,
        ssm_cfg=None,
        norm_epsilon: float = 1e-12,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        lookahead_kernel: int = 0,
        right_context: int = 0,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
    ):
        # assert typechecked()
        super().__init__()
        self._output_size = output_size
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32

        if input_layer == "causal_conv2d":
            self.embed = CausalConv2dSubsampling(input_size, output_size, (3, 3), bias=True)
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_size, dropout_rate)
        elif input_layer == "conv2d_m":
            self.embed = Conv2dSubsamplingM(input_size, output_size, dropout_rate)
        elif input_layer is None:
            if input_size == output_size:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_size)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.norm_before_mamba = LayerNorm(output_size)
        d_model = output_size
        n_layer = num_blocks
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # self.mamba_layer_dropout = torch.nn.Dropout(dropout_rate)

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

       
        self.lookahead_kernel = lookahead_kernel
        self.right_context = right_context
        self.left_context = lookahead_kernel - 1 - self.right_context

        if lookahead_kernel > 0:

            # self.lookahead_cnn0 = nn.Conv1d(
            #     output_size,
            #     output_size,
            #     lookahead_kernel,
            #     stride=1,
            #     padding=0,
            #     bias=True,
            # )

            self.lookahead_cnn = nn.Conv1d(
                output_size,
                output_size,
                lookahead_kernel,
                stride=1,
                padding=lookahead_kernel//2,
                bias=True,
            )

            activation_type = "swish"
            self.activation = get_activation(activation_type)

            self.lookahead_norm = LayerNorm(output_size)
            self.dropout = torch.nn.Dropout(dropout_rate)

            

        # output_size_embed = 256
        # self._output_size = output_size_embed
        
        if not hasattr(self, "lookahead_cnn"):
            self.encoder_out_embed = torch.nn.Sequential(
                    torch.nn.Linear(output_size, output_size),
                    get_activation('swish'),
                    LayerNorm(output_size),                                                                                            
                    torch.nn.Dropout(dropout_rate),
                )

        # self.mamba_dropout = torch.nn.Dropout(dropout_rate)
        # if self.normalize_before:
        #     self.after_norm = LayerNorm(output_size)

        
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) <= num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }
    
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
        inference_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        # initdtype = xs_pad.dtype
        # logging.info(f'xs_pad: {xs_pad.device}, {xs_pad.dtype}')
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsamplingM)
            or isinstance(self.embed, CausalConv2dSubsampling)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        if hasattr(self, "norm_before_mamba"):
            xs_pad = self.norm_before_mamba(xs_pad)
        # if hasattr(self, "lookahead_cnn"):
        #     xs_pad = self.lookahead_cnn(xs_pad.transpose(1, 2)).transpose(1, 2)
        #     xs_pad = self.activation(xs_pad)
        #     xs_pad = self.lookahead_norm(xs_pad)

        residual = None
        # for layer in self.layers:
        #     xs_pad, residual = layer(
        #         xs_pad, residual, inference_params=inference_params
        #     )
        # if not self.fused_add_norm:
        #     residual = (xs_pad + residual) if residual is not None else xs_pad
        #     xs_pad = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        # else:
        #     # Set prenorm=False here since we don't need the residual
        #     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        #     xs_pad = fused_add_norm_fn(
        #         xs_pad,
        #         self.norm_f.weight,
        #         self.norm_f.bias,
        #         eps=self.norm_f.eps,
        #         residual=residual,
        #         prenorm=False,
        #         residual_in_fp32=self.residual_in_fp32,
        #     )

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            for layer_idx, layer in enumerate(self.layers):
                xs_pad, residual = layer(xs_pad, residual, inference_params=inference_params)
                if hasattr(self, "mamba_layer_dropout"):
                    xs_pad = self.mamba_layer_dropout(xs_pad)
                # if layer_idx+1 == 8 and hasattr(self, "lookahead_cnn"):
                #     xs_pad = torch.nn.functional.pad(xs_pad, (0, 0, self.left_context, self.right_context))
                #     xs_pad = self.lookahead_cnn0(xs_pad.transpose(1, 2)).transpose(1, 2)
                #     # xs_pad = self.activation(xs_pad)     
                #     # xs_pad = self.dropout(xs_pad)
                #     xs_pad = self.lookahead_norm0(xs_pad)

        else:
            for layer_idx, layer in enumerate(self.layers):
                xs_pad, residual = layer(xs_pad, residual, inference_params=inference_params)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    if not self.fused_add_norm:
                        residual = (encoder_out + residual) if residual is not None else encoder_out
                        encoder_out = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
                    else:
                        # Set prenorm=False here since we don't need the residual
                        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                        encoder_out = fused_add_norm_fn(
                            encoder_out,
                            self.norm_f.weight,
                            self.norm_f.bias,
                            eps=self.norm_f.eps,
                            residual=residual,
                            prenorm=False,
                            residual_in_fp32=self.residual_in_fp32,
                            )

                    encoder_out = self.encoder_out_embed(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(xs_pad, tuple):
                            x, pos_emb = xs_pad
                            x = x + self.conditioning_layer(ctc_out)
                            xs_pad = (x, pos_emb)
                        else:
                            xs_pad = xs_pad + self.conditioning_layer(ctc_out) 
        
        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        
        if not self.fused_add_norm:
                residual = (xs_pad + residual) if residual is not None else xs_pad
                xs_pad = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            xs_pad = fused_add_norm_fn(
                xs_pad,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                )
        
        if hasattr(self, "depthwise_conv"):
            xs_pad = self.depthwise_conv(xs_pad.transpose(1, 2)).transpose(1, 2)
            xs_pad = self.depthwise_conv2(xs_pad.transpose(1, 2)).transpose(1, 2)
            xs_pad = self.activation(xs_pad)


        if hasattr(self, "lookahead_cnn"):
            # use dinamic padding
            # self.right_context = torch.randint(low=0, high=self.lookahead_kernel, size=(1,)).item()
            # self.right_context = 10
            # self.left_context = self.lookahead_kernel - 1 - self.right_context
            # xs_pad = torch.nn.functional.pad(xs_pad, (0, 0, self.left_context, self.right_context))
            xs_pad = self.lookahead_cnn(xs_pad.transpose(1, 2)).transpose(1, 2)
            xs_pad = self.activation(xs_pad)     
            xs_pad = self.lookahead_norm(xs_pad)
            xs_pad = self.dropout(xs_pad)
        
        if hasattr(self, "mamba_dropout"):
            xs_pad = self.mamba_dropout(xs_pad)

        if hasattr(self, "encoder_out_embed"):
            xs_pad = self.encoder_out_embed(xs_pad)
    
        # logging.info(f'xs_pad: {xs_pad.device}, {xs_pad.dtype}')
        # logging.info(f'xs_pad: {xs_pad.shape}')
        # xs_pad = xs_pad
        # logging.info(f'xs_pad: {xs_pad.shape}')
        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None

        return xs_pad, olens, None


