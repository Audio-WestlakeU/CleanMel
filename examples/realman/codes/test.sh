#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditors: FnoY0723 fangying@westlake.edu.cn
 # @LastEditTime: 2025-01-10 15:00:52
 # @FilePath: /InASR/examples/realman/codes/test.sh
### 
set -e
set -u

file="./networks/frontend/default_enh.py"
declare -a dirArray
mel_path="$1"
dirArray+=("$mel_path")
# dirArray+=("/data/home/fangying/sn_enh_mel/mels/8xSPB_Hid128_offline_real_rts_ensemble139-149")

# for d in $(ls -d "$enhPath"/*/); do
#     dirArray+=("$d")
# done

snr_level="$2"
if [ "$snr_level" == "highsnr" ]; then
    sed -i "149c \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ self.base_mels_path = os.path.join(self.base_mels_path, 'RealMAN_highsnr')" "$file"
elif [ "$snr_level" == "lowsnr" ]; then
    sed -i "149c \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ self.base_mels_path = os.path.join(self.base_mels_path, 'RealMAN')" "$file"
else
    echo "Invalid SNR level"
    exit 1
fi

for i in "${dirArray[@]}"; do
    sed -i "95c \ \ \ \ \ \ \ \ self.base_mels_path = \"$i\"" "$file"
    echo "$i"
    python ./examples/realman/codes/test.py --test_path "$i" --snr_level "$snr_level"
done
