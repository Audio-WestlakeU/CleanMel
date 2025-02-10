'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2025-02-10 10:17:51
FilePath: /InASR/examples/reverb/wpy_test_sim.py
'''
import os
import time

scp_file = "/data/home/fangying/InASR/examples/reverb/data/et_simu_1ch_wpy/wav.scp"
test_list=["NeGITCNSAS-v1-v1-100-sim"]
orgname = "NeGITCNSAS-v1-v1-100-sim"

for i,newname in enumerate(test_list):
    print(f"\nProcessing {i+1}/{len(test_list)}: {newname}")
    with open(scp_file, 'r') as f:
        lines = f.readlines()

    f = open(scp_file, 'w')
    for line in lines:
        line = line.replace(orgname, newname)
        f.write(line)
    f.close()
    
    time.sleep(1)
    os.system(f"CUDA_VISIBLE_DEVICES=5 ./examples/reverb/evaluate_via_whisper_sim.sh")
    time.sleep(1)

    if os.path.exists("./examples/reverb/whisper-medium_outputs/et_simu_1ch_wpy"):
        os.system(f'mv ./examples/reverb/whisper-medium_outputs/et_simu_1ch_wpy ./examples/reverb/whisper-medium_outputs/et_simu_1ch_{newname}')

    orgname = newname 