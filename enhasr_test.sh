#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-02-14 14:43:21
 # @FilePath: /InASR/enhasr_test.sh
### 
set -e
set -u

mel_path="$1"
echo "Processing ${mel_path}"

cd /data/home/fangying/InASR/

echo ""
echo "Testing on RealMan lowsnr"
./examples/realman/codes/test.sh "${mel_path}" "lowsnr"

echo ""
echo "Testing on CHiME4"
./examples/chime4/test.sh "${mel_path}"

echo ""
echo "Testing on REVERB"
./examples/reverb/test.sh "${mel_path}"

echo ""
echo "Testing on WenetSpeech"
./examples/wenetspeech/test.sh "${mel_path}"
 
# echo ""
# echo "Testing on RealMan highsnr"
# ./examples/realman/codes/test.sh "${mel_path}" "highsnr"

echo "All tests Done"