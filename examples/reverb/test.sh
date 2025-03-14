#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-03-05 15:56:56
 # @FilePath: /InASR/examples/reverb/test.sh
### 
set -e
set -u

expPath="./examples/reverb/asr_train_asr_transformer4_raw_en_char_sp/decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_10best_asr_model_valid.acc.ave_10best/et_real_1ch"
parentDir="$(dirname "$expPath")"
file="./networks/frontend/default_enh.py"
declare -a dirArray
mel_path="$1"
dirArray+=("$mel_path")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_real_rts_ensemble139-149")


for i in "${dirArray[@]}"; do
    sed -i "95c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    CUDA_VISIBLE_DEVICES=4 ./examples/reverb/run.sh --test_sets "et_real_1ch" --nj 48
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}_real"
    echo "$newName"
    mv "$expPath" "$newPath"
done 

expPath="./examples/reverb/asr_train_asr_transformer4_raw_en_char_sp/decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_10best_asr_model_valid.acc.ave_10best/et_simu_1ch"
parentDir="$(dirname "$expPath")"

for i in "${dirArray[@]}"; do
    sed -i "95c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    CUDA_VISIBLE_DEVICES=4 ./examples/reverb/run.sh --test_sets "et_simu_1ch" --nj 48
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}_simu"
    echo "$newName"
    mv "$expPath" "$newPath"
done

./examples/reverb/run.sh --test_sets "et_real_1ch et_simu_1ch" --nj 48 --stage 2
