#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-02-17 11:10:48
 # @FilePath: /InASR/examples/chime4/evaluate_via_whisper.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

whisper_tag=large-v2    # whisper model tag, e.g., small, medium, large, etc
cleaner=whisper_en
hyp_cleaner=whisper_en
nj=1
test_sets="et05_real_isolated_1ch_track et05_simu_isolated_1ch_track"
# decode_options is used in Whisper model's transcribe method
decode_options="{language: en, task: transcribe, temperature: 0, beam_size: 10, fp16: False}"

for x in ${test_sets}; do
    wavscp=./examples/chime4/data/${x}/wav.scp    # path to wav.scp
    outdir=./examples/chime4/whisper-${whisper_tag}_outputs/${x}  # path to save output
    gt_text=./examples/chime4/data/${x}/text      # path to groundtruth text file (for scoring only)

    utils/evaluate_asr.sh \
        --whisper_tag ${whisper_tag} \
        --nj ${nj} \
        --gpu_inference true \
        --inference_nj 8 \
        --stage 2 \
        --stop_stage 3 \
        --cleaner ${cleaner} \
        --hyp_cleaner ${hyp_cleaner} \
        --decode_options "${decode_options}" \
        --gt_text ${gt_text} \
        ${wavscp} \
        ${outdir}
done