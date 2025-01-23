#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2024-10-22 15:37:26
 # @FilePath: /InASR/examples/chime4/test.sh
### 
set -e
set -u

enhPath="/data/home/fangying/sn_enh_mel"
expPath="./examples/chime4/asr_train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k_raw_en_char_sp/decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave_10best/et05_real_isolated_1ch_track"
parentDir="$(dirname "$expPath")"
file="./networks/frontend/default_enh.py"
declare -a dirArray
mel_path="$1"
dirArray+=("$mel_path")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_real_rts_ensemble139-149")


for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    CUDA_VISIBLE_DEVICES=5 ./examples/chime4/run.sh --test_sets "et05_real_isolated_1ch_track" --nj 48
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}_real"
    echo "$newName"
    mv "$expPath" "$newPath"
done 

expPath="./examples/chime4/asr_train_asr_e_branchformer_e10_mlp1024_linear1024_macaron_lr1e-3_warmup25k_raw_en_char_sp/decode_asr_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave_10best/et05_simu_isolated_1ch_track"
parentDir="$(dirname "$expPath")"

for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    CUDA_VISIBLE_DEVICES=5 ./examples/chime4/run.sh --test_sets "et05_simu_isolated_1ch_track" --nj 48
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}_simu"
    echo "$newName"
    mv "$expPath" "$newPath"
done

./examples/chime4/run.sh --test_sets "et05_real_isolated_1ch_track et05_simu_isolated_1ch_track" --nj 48 --stage 2
