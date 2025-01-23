#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-01-10 14:37:49
 # @FilePath: /InASR/examples/wenetspeech/test.sh
### 
set -e
set -u

enhPath="/data/home/fangying/sn_enh_mel"
expPath="./examples/wenetspeech/asr_train_asr_raw_zh_char/decode_asr_asr_model_valid.acc.ave_10best/test_meeting"
parentDir="$(dirname "$expPath")"
file="./networks/frontend/default_enh.py"
declare -a dirArray
mel_path="$1"
dirArray+=("$mel_path")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_real_rts_ensemble139-149")

for i in "${dirArray[@]}"; do
    sed -i "108c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "Mel_path is $i"
    ./examples/wenetspeech/run.sh
    newName=$(basename "$i")
    newPath="${parentDir}/${newName}"
    echo "$newName"
    mv "$expPath" "$newPath"
done 

./examples/wenetspeech/run.sh --stage 2
