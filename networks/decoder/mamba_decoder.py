# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn
try:
    from networks.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from networks.mamba_ssm.modules.mamba_simple import Mamba, Block
# from typeguard import typechecked
from networks.decoder.abs_decoder import AbsDecoder
from utils.nets_utils import make_pad_mask
from networks.layer_norm import LayerNorm
from networks.ctc import CTC

import logging
import math
from functools import partial
from utils.nets_utils import get_activation



def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
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


class MambaDecoder(AbsDecoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        output_size: int = 512,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        positional_dropout_rate: float = 0.1,
    ):
        # assert typechecked()
        super().__init__()
        self._output_size = output_size

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, output_size),
            LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            get_activation("swish"),
            # PositionalEncoding(encoder_output_size, positional_dropout_rate),
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.normalize_before = normalize_before
        
        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

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
        # 
        # self.mamba_dropout = torch.nn.Dropout(dropout_rate)

        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

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
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        ctc: CTC = None,
        inference_params=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """

        masks = (~make_pad_mask(hlens)[:, None, :]).to(hs_pad.device)

        hs_pad = self.embed(hs_pad)

        residual = None
        for layer in self.layers:
            hs_pad, residual = layer(hs_pad, residual, inference_params=inference_params)

            if hasattr(self, "mamba_layer_dropout"):
                hs_pad = self.mamba_layer_dropout(hs_pad)

        if not self.fused_add_norm:
            residual = (hs_pad + residual) if residual is not None else hs_pad
            hs_pad = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hs_pad = fused_add_norm_fn(
                hs_pad,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # intermediate_outs = []
        # if len(self.interctc_layer_idx) == 0:
        #     hs_pad, masks = self.encoders(hs_pad, masks)
        # else:
        #     for layer_idx, encoder_layer in enumerate(self.encoders):
        #         hs_pad, masks = encoder_layer(hs_pad, masks)

        #         if layer_idx + 1 in self.interctc_layer_idx:
        #             encoder_out = hs_pad
        #             if isinstance(encoder_out, tuple):
        #                 encoder_out = encoder_out[0]

        #             # intermediate outputs are also normalized
        #             if self.normalize_before:
        #                 encoder_out = self.after_norm(encoder_out)

        #             intermediate_outs.append((layer_idx + 1, encoder_out))

        #             if self.interctc_use_conditioning:
        #                 ctc_out = ctc.softmax(encoder_out)

        #                 if isinstance(hs_pad, tuple):
        #                     x, pos_emb = hs_pad
        #                     x = x + self.conditioning_layer(ctc_out)
        #                     hs_pad = (x, pos_emb)
        #                 else:
        #                     hs_pad = hs_pad + self.conditioning_layer(ctc_out) 

        if isinstance(hs_pad, tuple):
            hs_pad = hs_pad[0]
        
        # hs_pad = self.mamba_dropout(hs_pad)

        if self.normalize_before:
            hs_pad = self.after_norm(hs_pad)

        olens = masks.squeeze(1).sum(1)

        # if len(intermediate_outs) > 0:
        #     return (hs_pad, intermediate_outs), olens
        
        return hs_pad, olens


