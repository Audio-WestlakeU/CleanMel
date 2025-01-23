# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any, List, Sequence, Tuple

import torch
# from typeguard import typechecked

from networks.decoder.abs_decoder import AbsDecoder
from utils.nets_utils import make_pad_mask
from networks.transformer.attention import MultiHeadedAttention
from networks.transformer.attention import StreamingMultiHeadedAttention, ChunkMultiHeadedAttention, LookaheadMultiHeadedAttention
from networks.transformer.embedding import PositionalEncoding
from networks.transformer.encoder_layer import EncoderLayer, ChunkEncoderLayer
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

from networks.ctc import CTC
from utils.nets_utils import get_activation

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
    # use_dynamic_chunk = False
    # chunk_size = 5
    if use_dynamic_chunk:
        max_len = size
        chunk_size = torch.randint(1, max_len, (1, )).item()
        if chunk_size > max_len // 2:
                chunk_size = max_len
        else:
            chunk_size = chunk_size % 5 + 1

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

class UnimodalAttentionDecoder(AbsDecoder):
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
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        lookahead_kernel: int = 0,
        right_context: int = 0,
        chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
    ):
        # assert typechecked()
        super().__init__()
        self._output_size = output_size

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, output_size),
            LayerNorm(output_size),
            torch.nn.Dropout(dropout_rate),
            get_activation("swish"),
            pos_enc_class(output_size, positional_dropout_rate),
        )
        
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
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
        
        self.chunk_size = chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk

        if self.chunk_size > 0:
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    MultiHeadedAttention(
                        attention_heads, output_size, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else:
            self.encoders = repeat(
                num_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    StreamingMultiHeadedAttention(
                        attention_heads, output_size, attention_dropout_rate
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )

        # self.la_decoder = EncoderLayer(
        #         output_size,
        #         LookaheadMultiHeadedAttention(
        #             attention_heads, output_size, attention_dropout_rate
        #         ),
        #         positionwise_layer(*positionwise_layer_args),
        #         dropout_rate,
        #         normalize_before,
        #         concat_after,
        #     )
        

        self.lookahead_kernel = lookahead_kernel
        self.right_context = right_context
        self.left_context = lookahead_kernel - 1 - self.right_context
        if lookahead_kernel > 0:
            # self.lookahead_cnn = torch.nn.Conv1d(
            #     output_size,
            #     output_size,
            #     lookahead_kernel,
            #     stride=1,
            #     padding=0,
            #     bias=True,
            # )

            self.lookahead_cnn = torch.nn.Conv1d(
                output_size,
                output_size,
                lookahead_kernel,
                stride=1,
                padding= lookahead_kernel // 2,
                bias=True,
            )

            activation_type = "swish"
            self.activation = get_activation(activation_type)

            self.lookahead_norm = LayerNorm(output_size)
            self.dropout = torch.nn.Dropout(dropout_rate)


        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) <= num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None
        if self.interctc_use_conditioning:
            self.conditioning_act = get_activation("swish")
            self.conditioning_cnn = torch.nn.Conv1d(output_size, output_size, 3, stride=1, padding=1, bias=True,)


    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """

        masks = (~make_pad_mask(hlens)[:, None, :]).to(hs_pad.device)

        hs_pad = self.embed(hs_pad)

        if self.chunk_size > 0:
            chunk_masks = make_chunk_mask(hs_pad.size(1), self.chunk_size, self.use_dynamic_chunk,device=hs_pad.device) # (L, L)
            chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
            # logging.info(f'{chunk_masks[0]}')
            chunk_masks = masks & chunk_masks  # (B, L, L)

        # for i, encoder_layer in enumerate(self.la_encoder):
        #     hs_pad, masks = encoder_layer(hs_pad, masks)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            if self.chunk_size > 0:
                # hs_pad, masks = self.encoders(hs_pad, masks)
                hs_pad, chunk_masks = self.encoders(hs_pad, chunk_masks)
            else:
                for layer_idx, encoder_layer in enumerate(self.encoders):
                    hs_pad, masks = encoder_layer(hs_pad, masks)
                    if layer_idx + 1 == 4:
                        if hasattr(self, "lookahead_cnn"):
                            # hs_pad = torch.nn.functional.pad(hs_pad, (0, 0, self.left_context, self.right_context))
                            hs_pad = self.lookahead_cnn(hs_pad.transpose(1, 2)).transpose(1, 2)
                            hs_pad = self.activation(hs_pad)     
                            hs_pad = self.lookahead_norm(hs_pad)
                            hs_pad = self.dropout(hs_pad)

        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                hs_pad, masks = encoder_layer(hs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = hs_pad
                    if isinstance(encoder_out, tuple):
                        encoder_out = encoder_out[0]

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)

                        if isinstance(hs_pad, tuple):
                            x, pos_emb = hs_pad
                            condition = self.conditioning_layer(ctc_out)
                            condition = self.conditioning_act(condition)
                            condition = self.conditioning_cnn(condition.transpose(1, 2)).transpose(1, 2)
                            x = x + condition
                            hs_pad = (x, pos_emb)
                        else:
                            condition = self.conditioning_layer(ctc_out)
                            condition = self.conditioning_act(condition)
                            condition = self.conditioning_cnn(condition.transpose(1, 2)).transpose(1, 2)
                            hs_pad = hs_pad + condition 

                        # if isinstance(hs_pad, tuple):
                        #     x, pos_emb = hs_pad
                        #     x = x + self.conditioning_layer(ctc_out)
                        #     hs_pad = (x, pos_emb)
                        # else:
                        #     hs_pad = hs_pad + self.conditioning_layer(ctc_out) 
        
        # if hasattr(self, "la_decoder"):
        #     hs_pad = self.la_decoder(hs_pad, masks)
        if isinstance(hs_pad, tuple):
            hs_pad = hs_pad[0]
        
        # if hasattr(self, "lookahead_cnn"):
        #     # hs_pad = torch.nn.functional.pad(hs_pad, (0, 0, self.left_context, self.right_context))
        #     xs_pad = self.lookahead_cnn(xs_pad.transpose(1, 2)).transpose(1, 2)
        #     xs_pad = self.activation(xs_pad)     
        #     xs_pad = self.lookahead_norm(xs_pad)
        #     xs_pad = self.dropout(xs_pad)


        if self.normalize_before:
            hs_pad = self.after_norm(hs_pad)

        olens = masks.squeeze(1).sum(1)

        if len(intermediate_outs) > 0:
            return (hs_pad, intermediate_outs), olens
        
        return hs_pad, olens

# class UnimodalAttentionDecoder(AbsDecoder):
#     """Transformer encoder module.

#     Args:
#         input_size: input dim
#         output_size: dimension of attention
#         attention_heads: the number of heads of multi head attention
#         linear_units: the number of units of position-wise feed forward
#         num_blocks: the number of decoder blocks
#         dropout_rate: dropout rate
#         attention_dropout_rate: dropout rate in attention
#         positional_dropout_rate: dropout rate after adding positional encoding
#         input_layer: input layer type
#         pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
#         normalize_before: whether to use layer_norm before the first block
#         concat_after: whether to concat attention layer's input and output
#             if True, additional linear will be applied.
#             i.e. x -> x + linear(concat(x, att(x)))
#             if False, no additional linear will be applied.
#             i.e. x -> x + att(x)
#         positionwise_layer_type: linear of conv1d
#         positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
#         padding_idx: padding_idx for input_layer=embed
#     """

#     def __init__(
#         self,
#         vocab_size: int,
#         encoder_output_size: int,
#         output_size: int = 256,
#         attention_heads: int = 4,
#         linear_units: int = 2048,
#         num_blocks: int = 6,
#         dropout_rate: float = 0.1,
#         positional_dropout_rate: float = 0.1,
#         attention_dropout_rate: float = 0.0,
#         pos_enc_class=PositionalEncoding,
#         normalize_before: bool = True,
#         concat_after: bool = False,
#         positionwise_layer_type: str = "linear",
#         positionwise_conv_kernel_size: int = 1,
#         padding_idx: int = -1,
#         interctc_layer_idx: List[int] = [],
#         interctc_use_conditioning: bool = False,
#     ):
#         # assert typechecked()
#         super().__init__()
#         self._output_size = output_size

#         self.embed = torch.nn.Sequential(
#             torch.nn.Linear(encoder_output_size, output_size),
#             torch.nn.LayerNorm(output_size),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.ReLU(),
#             pos_enc_class(output_size, positional_dropout_rate),
#         )
        
#         self.normalize_before = normalize_before
#         if positionwise_layer_type == "linear":
#             positionwise_layer = PositionwiseFeedForward
#             positionwise_layer_args = (
#                 output_size,
#                 linear_units,
#                 dropout_rate,
#             )
#         elif positionwise_layer_type == "conv1d":
#             positionwise_layer = MultiLayeredConv1d
#             positionwise_layer_args = (
#                 output_size,
#                 linear_units,
#                 positionwise_conv_kernel_size,
#                 dropout_rate,
#             )
#         elif positionwise_layer_type == "conv1d-linear":
#             positionwise_layer = Conv1dLinear
#             positionwise_layer_args = (
#                 output_size,
#                 linear_units,
#                 positionwise_conv_kernel_size,
#                 dropout_rate,
#             )
#         else:
#             raise NotImplementedError("Support only linear or conv1d.")
#         self.encoders = repeat(
#             num_blocks,
#             lambda lnum: ChunkEncoderLayer(
#                 output_size,
#                 ChunkMultiHeadedAttention(
#                     attention_heads, output_size, attention_dropout_rate
#                 ),
#                 # StreamingMultiHeadedAttention(
#                 #     attention_heads, output_size, attention_dropout_rate
#                 # ),
#                 positionwise_layer(*positionwise_layer_args),
#                 dropout_rate,
#                 normalize_before,
#                 concat_after,
#             ),
#         )
#         if self.normalize_before:
#             self.after_norm = LayerNorm(output_size)

#         self.interctc_layer_idx = interctc_layer_idx
#         if len(interctc_layer_idx) > 0:
#             assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
#         self.interctc_use_conditioning = interctc_use_conditioning
#         self.conditioning_layer = None


#     def output_size(self) -> int:
#         return self._output_size

#     def forward(
#         self,
#         hs_pad: torch.Tensor,
#         hlens: torch.Tensor,
#         ys_in_pad: torch.Tensor,
#         ys_in_lens: torch.Tensor,
#         ctc: CTC = None,
#         chunk_counts: torch.Tensor = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Embed positions in tensor.

#         Args:
#             xs_pad: input tensor (B, L, D)
#             ilens: input length (B)
#             prev_states: Not to be used now.
#         Returns:
#             position embedded tensor and mask
#         """

#         masks = (~make_pad_mask(hlens)[:, None, :]).to(hs_pad.device)
#         hs_pad = self.embed(hs_pad)

#         intermediate_outs = []
#         if len(self.interctc_layer_idx) == 0:
#             hs_pad, masks, _ = self.encoders(hs_pad, masks, chunk_counts)
#         else:
#             for layer_idx, decoder_layer in enumerate(self.encoders):
#                 hs_pad, masks, _ = decoder_layer(hs_pad, masks, chunk_counts)

#                 if layer_idx + 1 in self.interctc_layer_idx:
#                     decoder_out = hs_pad
#                     if isinstance(decoder_out, tuple):
#                         decoder_out = decoder_out[0]

#                     # intermediate outputs are also normalized
#                     if self.normalize_before:
#                         decoder_out = self.after_norm(decoder_out)

#                     intermediate_outs.append((layer_idx + 1, decoder_out))

#                     if self.interctc_use_conditioning:
#                         ctc_out = ctc.softmax(decoder_out)

#                         if isinstance(hs_pad, tuple):
#                             x, pos_emb = hs_pad
#                             x = x + self.conditioning_layer(ctc_out)
#                             hs_pad = (x, pos_emb)
#                         else:
#                             hs_pad = hs_pad + self.conditioning_layer(ctc_out) 

#         if isinstance(hs_pad, tuple):
#             hs_pad = hs_pad[0]
        
#         if self.normalize_before:
#             hs_pad = self.after_norm(hs_pad)

#         olens = masks.squeeze(1).sum(1)

#         if len(intermediate_outs) > 0:
#             return (hs_pad, intermediate_outs), olens
        
#         return hs_pad, olens
