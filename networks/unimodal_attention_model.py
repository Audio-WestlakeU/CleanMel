'''
Author: FnoY fangying@westlake.edu.cn
LastEditTime: 2024-09-30 14:49:44
FilePath: /espnet/espnet2/asr/unimodal_attention_model.py
'''
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
# from typeguard import typechecked

from networks.uma import UMA
from networks.ctc import CTC
from networks.decoder.abs_decoder import AbsDecoder
from networks.encoder.abs_encoder import AbsEncoder
from networks.frontend.abs_frontend import AbsFrontend
from networks.postencoder.abs_postencoder import AbsPostEncoder
from networks.preencoder.abs_preencoder import AbsPreEncoder
from networks.specaug.abs_specaug import AbsSpecAug
from networks.transducer.error_calculator import ErrorCalculatorTransducer
from networks.transducer.utils import get_transducer_task_io
from networks.layers.abs_normalize import AbsNormalize
from utils.device_funcs import force_gatherable
from tasks.abs_espnet_model import AbsESPnetModel
from search.e2e_asr_common import ErrorCalculator
from utils.nets_utils import th_accuracy
from networks.transformer.add_sos_eos import add_sos_eos
from networks.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

from matplotlib import pyplot as plt

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class UAMASRModel(AbsESPnetModel):
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: torch.nn.Module,
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        uma: UMA,
        joint_network: Optional[torch.nn.Module],
        aux_ctc: dict = None,
        ctc_weight: float = 0.5,
        interctc_weight_enc: float = 0.0,
        interctc_weight_dec: float = 0.0,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        transducer_multi_blank_durations: List = [],
        transducer_multi_blank_sigma: float = 0.05,
        # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
        # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
        sym_sos: str = "<sos/eos>",
        sym_eos: str = "<sos/eos>",
        extract_feats_in_collect_stats: bool = True,
        lang_token_id: int = -1,
        enc_ctc_weight: float = 0.0,
    ):
        # assert typechecked()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= interctc_weight_enc < 1.0, interctc_weight_enc
        assert 0.0 <= interctc_weight_dec < 1.0, interctc_weight_dec

        super().__init__()
        # NOTE (Shih-Lun): else case is for OpenAI Whisper ASR model,
        #                  which doesn't use <blank> token
        if sym_blank in token_list:
            self.blank_id = token_list.index(sym_blank)
        else:
            self.blank_id = 0
        if sym_sos in token_list:
            self.sos = token_list.index(sym_sos)
        else:
            self.sos = vocab_size - 1
        if sym_eos in token_list:
            self.eos = token_list.index(sym_eos)
        else:
            self.eos = vocab_size - 1
        
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.interctc_weight_enc = interctc_weight_enc
        self.interctc_weight_dec = interctc_weight_dec

        self.aux_ctc = aux_ctc
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if not hasattr(self.encoder, "interctc_use_conditioning"):
                self.encoder.interctc_use_conditioning = False
        if self.encoder.interctc_use_conditioning:
            self.encoder.conditioning_layer = torch.nn.Linear(
                vocab_size, self.encoder.output_size()
            )

        self.uma = uma

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            self.decoder = decoder
            self.joint_network = joint_network

            if not transducer_multi_blank_durations:
                from warprnnt_pytorch import RNNTLoss

                self.criterion_transducer = RNNTLoss(
                    blank=self.blank_id,
                    fastemit_lambda=0.0,
                )
            else:
                from networks.transducer.rnnt_multi_blank.rnnt_multi_blank import (
                    MultiblankRNNTLossNumba,
                )

                self.criterion_transducer = MultiblankRNNTLossNumba(
                    blank=self.blank_id,
                    big_blank_durations=transducer_multi_blank_durations,
                    sigma=transducer_multi_blank_sigma,
                    reduction="mean",
                    fastemit_lambda=0.0,
                )
                self.transducer_multi_blank_durations = transducer_multi_blank_durations

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            # if ctc_weight < 1.0:
            #     assert (
            #         decoder is not None
            #     ), "decoder should not be None when attention is used"
            # else:
            #     decoder = None
            #     logging.warning("Set decoder to none as ctc_weight==1.0")

            self.decoder = decoder

            # if not hasattr(self.decoder, "interctc_use_conditioning"):
            #     self.decoder.interctc_use_conditioning = False
                
            # if self.decoder.interctc_use_conditioning:
            #     self.decoder.conditioning_layer = torch.nn.Linear(
            #         vocab_size, self.decoder.output_size()
            #     )

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        self.is_encoder_whisper = "Whisper" in type(self.encoder).__name__

        if self.is_encoder_whisper:
            assert (
                self.frontend is None
            ), "frontend should be None when using full Whisper model"

        if lang_token_id != -1:
            self.lang_token_id = torch.tensor([[lang_token_id]])
        else:
            self.lang_token_id = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
            kwargs: "utt_id" is among the input.
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        # print("encoder_out_length: ", encoder_out_lens)
        # print("encoder_out: ", encoder_out)
        intermediate_outs_enc = None
        if isinstance(encoder_out, tuple):
            intermediate_outs_enc = encoder_out[1]
            encoder_out = encoder_out[0]

        loss_ctc, cer_ctc = None, None
        loss_enc_ctc, cer_enc_ctc = None, None
        loss_blank = None
        loss_pfr = None
        loss_fast = None
        uma_reduction = None
        text_vs_uma = None

        # gaussian_w, gaussian_b = None, None
        stats = dict()
        
        # 3. unimodal attention module
        uma_out, uma_out_lens, chunk_counts = self.uma(encoder_out, encoder_out_lens)
        stats["uma_reduction"] = (uma_out_lens.sum().item())/(encoder_out_lens.sum().item())
        stats["text_vs_uma"] = (text_lengths.sum().item())/(uma_out_lens.sum().item())
        # stats["gaussian_w"] = gaussian_w
        # stats["gaussian_b"] = gaussian_b
        # logging.info("uma_out_length: "+ (str(uma_out_lens)))

        # 2. Decoder
        if self.decoder is not None:
            decoder_out, decoder_out_lens = self.decoder(uma_out, uma_out_lens, text, text_lengths, self.ctc)
            # decoder_out, decoder_out_lens = self.decoder(uma_out, uma_out_lens, text, text_lengths, self.ctc, chunk_counts)
            # print("decoder_out: ", decoder_out_lens)
            intermediate_outs_dec = None
            if isinstance(decoder_out, tuple):
                intermediate_outs_dec = decoder_out[1]
                decoder_out = decoder_out[0]
        else:
            decoder_out=uma_out
            decoder_out_lens=uma_out_lens


        # 1. CTC branch
        if self.ctc_weight != 0.0:
            # if stats["text_vs_uma"]>=1.0:
            #     ys_hat = self.ctc.ctc_ys_hat(decoder_out)
            #     loss_ctc = self.criterion_att(ys_hat, text)
            # else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                decoder_out, decoder_out_lens, text, text_lengths
            )

            # loss_enc_ctc, cer_enc_ctc = self._calc_ctc_loss(
            #     encoder_out, encoder_out_lens, text, text_lengths
            # )
            # stats["loss_enc_ctc"] = loss_enc_ctc.detach() if loss_enc_ctc is not None else None
            # stats["cer_enc_ctc"] = cer_enc_ctc
            

            # ys_prob = -self.ctc.log_softmax(decoder_out)
            # ys_idx = self.ctc.argmax(decoder_out)
            
            # sub_idx = ys_idx[:, 1:]
            # zero_column = torch.zeros((ys_idx.shape[0], 1), dtype=ys_idx.dtype).to(ys_idx.device)
            # penalty_idx = torch.cat((sub_idx, zero_column), dim=1)
            # penalty_result = torch.zeros(ys_prob.shape, dtype=penalty_idx.dtype).to(penalty_idx.device)
            # penalty_result.scatter_(2, penalty_idx.unsqueeze(-1), 1)
            # penalty_result[:,:,0] = 0
            
            # summed_values = penalty_result * ys_prob
            # summed_values = summed_values
    
            # sum_along_V = torch.sum(summed_values, dim=-1).masked_fill(~(ys_idx==0), 0)
            # sum_along_T = torch.sum(sum_along_V, dim=-1)
            # loss_fast = torch.mean(sum_along_T)*0.00001
            # stats["loss_fast"] = loss_fast.detach() if loss_fast is not None else None


            # ys_hat = self.ctc.ctc_ys_hat(decoder_out)
            # # ys_hat = self.ctc.ctc_ys_hat(encoder_out)
            # ys_hat = torch.nn.functional.log_softmax(ys_hat/10, dim=-1)
            # left_probs = ys_hat[:,:-1,:]
            # right_probs = ys_hat[:,1:,:]
            # loss_pfr = torch.nn.functional.kl_div(left_probs, right_probs, reduction="none", log_target=True)
            # mask = (make_pad_mask(decoder_out_lens)[:, :, None]).to(decoder_out.device)
            # loss_pfr.masked_fill_(mask[:,1:], 0)
            # loss_pfr = torch.sum(loss_pfr)/decoder_out.shape[0]*0.1
            # stats["loss_pfr"] = loss_pfr.detach() if loss_pfr is not None else None
            
            # masks = (make_pad_mask(decoder_out_lens)[:, :, None]).to(decoder_out.device)
            # ys_hat = self.ctc.softmax(decoder_out).masked_fill(masks, 0)
            # loss_blank = torch.sum(ys_hat[:,:,0])/torch.sum(decoder_out_lens)
            # stats["loss_blank"] = loss_blank.detach() if loss_blank is not None else None

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc
            stats["cer"] = cer_ctc


        # Encoder Intermediate CTC (optional)
        loss_interctc_enc = 0.0
        if self.interctc_weight_enc != 0.0 and intermediate_outs_enc is not None:
            for layer_idx, intermediate_out in intermediate_outs_enc:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                decoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, encoder_out_lens, text, text_lengths
                    )
                loss_interctc_enc = loss_interctc_enc + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_enclayer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_enclayer{}".format(layer_idx)] = cer_ic

            loss_interctc_enc = loss_interctc_enc / len(intermediate_outs_enc)

        # Decoder Intermediate CTC (optional)
        loss_interctc_dec = 0.0
        if self.interctc_weight_dec != 0.0 and intermediate_outs_dec is not None:
            for layer_idx, intermediate_out in intermediate_outs_dec:
                # we assume intermediate_out has the same length & padding
                # as those of encoder_out

                # use auxillary ctc data if specified
                loss_ic = None
                if self.aux_ctc is not None:
                    idx_key = str(layer_idx)
                    if idx_key in self.aux_ctc:
                        aux_data_key = self.aux_ctc[idx_key]
                        aux_data_tensor = kwargs.get(aux_data_key, None)
                        aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

                        if aux_data_tensor is not None and aux_data_lengths is not None:
                            loss_ic, cer_ic = self._calc_ctc_loss(
                                intermediate_out,
                                decoder_out_lens,
                                aux_data_tensor,
                                aux_data_lengths,
                            )
                        else:
                            raise Exception(
                                "Aux. CTC tasks were specified but no data was found"
                            )
                if loss_ic is None:
                    loss_ic, cer_ic = self._calc_ctc_loss(
                        intermediate_out, decoder_out_lens, text, text_lengths
                    ) 
                loss_interctc_dec = loss_interctc_dec + loss_ic

                # Collect Intermedaite CTC stats
                stats["loss_interctc_declayer{}".format(layer_idx)] = (
                    loss_ic.detach() if loss_ic is not None else None
                )
                stats["cer_interctc_declayer{}".format(layer_idx)] = cer_ic

            loss_interctc_dec = loss_interctc_dec / len(intermediate_outs_dec)

        # calculate whole intermediate loss
        loss = (
            1 - self.interctc_weight_enc - self.interctc_weight_dec
        ) * loss_ctc + self.interctc_weight_enc * loss_interctc_enc + self.interctc_weight_dec * loss_interctc_dec

        if loss_enc_ctc is not None:
            loss = loss*0.7 + loss_enc_ctc*0.3
        
        if loss_fast is not None:
            loss = loss + loss_fast
        
        if loss_pfr is not None:
            loss = loss + loss_pfr
        
        if loss_blank is not None:
            loss = loss + loss_blank
            
        # Collect total loss stats
        stats["loss"] = loss.detach()
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # logging.info("feats_length: "+str(feats_lengths))
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        # print("transformer encoder_out: ", encoder_out.shape)
        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )
        # print("postencoder_out: ", encoder_out.shape)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, :  speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        if hasattr(self, "lang_token_id") and self.lang_token_id is not None:
            ys_pad = torch.cat(
                [
                    self.lang_token_id.repeat(ys_pad.size(0), 1).to(ys_pad.device),
                    ys_pad,
                ],
                dim=1,
            )
            ys_pad_lens += 1

        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer

    def _calc_batch_ctc_loss(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        if self.ctc is None:
            return
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        # Calc CTC loss
        do_reduce = self.ctc.reduce
        self.ctc.reduce = False
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        self.ctc.reduce = do_reduce
        return loss_ctc