# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
# from typeguard import typechecked

from networks.ctc import CTC
from networks.encoder.abs_encoder import AbsEncoder
from networks.conformer.convolution import ConvolutionModule
from networks.conformer.encoder_layer import EncoderLayer
from utils.nets_utils import get_activation, make_pad_mask
from networks.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from networks.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from networks.transformer.layer_norm import LayerNorm
from networks.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from networks.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from networks.transformer.repeat import repeat
from networks.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
import torch.nn as nn

class CausalConv1dSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(CausalConv1dSubsampling, self).__init__()
        self.padding = (kernel_size - 1)

        self.subsample1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=2,
                bias=bias,
            ),
            nn.ReLU(),
        )

        self.subsample2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=2,
                # groups=out_channels,
                bias=bias,
            ),
            nn.ReLU(),
        )

    def forward(self, x, x_mask):
        x = x.transpose(1, 2)
        x = self.subsample1(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        x = self.subsample2(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        x = x.transpose(1, 2)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class CausalConv2dSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate, pos_enc=None):
        super(CausalConv2dSubsampling, self).__init__()
        self.padding = (kernel_size[0] - 1, 0) 

        self.subsample1 = nn.Sequential(
            nn.Conv2d(
                1, 
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                stride=2,
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
            ),
            nn.ReLU(),
        )

        # self.out = torch.nn.Linear(out_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(out_channels * (((in_channels - 1) // 2 - 1) // 2), out_channels),
            pos_enc if pos_enc is not None else PositionalEncoding(out_channels, dropout_rate),
        )

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


def make_chunk_mask(
        size: int,
        chunk_size: int,
        use_dynamic_chunk: bool,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    use_dynamic_chunk = False
    chunk_size = 20
    if use_dynamic_chunk:
        max_len = size
        chunk_size = torch.randint(1, max_len, (1, )).item()
        if chunk_size > max_len // 2:
                chunk_size = max_len
        else:
            chunk_size = chunk_size % 32 + 1

    # logging.info(f'chunk_size: {chunk_size}')
    # logging.info(f'use_dynamic_chunk: {use_dynamic_chunk}, chunk_size: {chunk_size}')
    org = torch.arange(size).repeat(size, 1).to(device)
    # ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    chunk_idx = torch.arange(0, size, chunk_size, device=device)
    chunk_idx = torch.cat((chunk_idx, torch.tensor([size], device=device)), dim=0)
    chunk_length = chunk_idx[1:] - chunk_idx[:-1]
    repeats = torch.repeat_interleave(chunk_idx[1:], chunk_length)
    ret = org < repeats.reshape(-1, 1)
    # ret = torch.tril(torch.ones(size, size), diagonal=0).bool().to(device)

    return ret


class ConformerChunkEncoder(AbsEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        layer_drop_rate: float = 0.0,
        max_pos_emb_len: int = 5000,
        chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        cnn_module_norm: str = "batch_norm",
        causal: bool = True,
    ):
        # assert typechecked()
        super().__init__()
        self._output_size = output_size

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)


        if input_layer == "causal_conv1d":
            self.embed = CausalConv1dSubsampling(input_size, output_size, 3, bias=True)
        elif input_layer == "causal_conv2d":
            self.embed = CausalConv2dSubsampling(
                input_size, 
                output_size, 
                (3, 3), 
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
                )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate, max_pos_emb_len)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum],
            ),
            layer_drop_rate,
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        self.chunk_size = chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, CausalConv1dSubsampling)
            or isinstance(self.embed, CausalConv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling)
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

        chunk_masks = make_chunk_mask(xs_pad[0].size(1), self.chunk_size, self.use_dynamic_chunk,device=xs_pad[0].device) # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        # logging.info(f'{chunk_masks[0]}')
        chunk_masks = masks & chunk_masks  # (B, L, L)
        # logging.info(f'{chunk_masks[0]}')
        # logging.info(f'chunk_masks: {chunk_masks.shape} {chunk_masks.dtype} {chunk_masks.device}')
        # logging.info(f'masks: {masks.shape} {masks.dtype} {masks.device}')

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, chunk_masks = self.encoders(xs_pad, chunk_masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, chunk_masks = encoder_layer(xs_pad, chunk_masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

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
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
    
        return xs_pad, olens, None
