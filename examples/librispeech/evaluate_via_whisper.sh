#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-02-17 11:10:48
 # @FilePath: /InASR/examples/librispeech/evaluate_via_whisper.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export OMP_NUM_THREADS=2

whisper_tag=base    # whisper model tag, e.g., small, medium, large, etc
cleaner=whisper_en
hyp_cleaner=whisper_en
nj=1
test_sets="test_clean"
# decode_options is used in Whisper model's transcribe method
decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

for x in ${test_sets}; do
    wavscp=./examples/librispeech/data/${x}/wav.scp    # path to wav.scp
    outdir=./examples/librispeech/whisper-${whisper_tag}_outputs/${x}  # path to save output
    gt_text=./examples/librispeech/data/${x}/text      # path to groundtruth text file (for scoring only)

    utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --nj ${nj} \
        --gpu_inference false \
        --inference_nj 40 \
        --stage 2 \
        --stop_stage 3 \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --decode_options "${decode_options}" \
        --gt_text ${gt_text} \
        ${wavscp} \
        ${outdir}
done