'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-11-27 15:47:48
FilePath: /InASR/examples/realman/codes/test.py
'''
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default=None, help='Path to the test data')
parser.add_argument('--snr_level', type=str, default=None, help='HighSNR or LowSNR')
parser.add_argument('--skip_inference', type=bool, default=False, help='If skip inference, only calculate CER')
args = parser.parse_args()

test_name = args.test_path.split('/')[-1]
if args.snr_level == 'highsnr':
    test_name = f'highsnr_{test_name}'
else:
    test_name = f'{test_name}'

# 
json_path = f"./examples/realman/results/{test_name}.json"

print(f'skip_inference: {args.skip_inference}')
if not args.skip_inference:
    os.system(f"python ./examples/realman/codes/0_inference.py --json_file {json_path}")
    print("Inference Done!\n")

os.system(f"python ./examples/realman/codes/1_convert_json_to_trn.py --json_file {json_path}")
print("Conversion Done!\n")

os.system(f"python ./examples/realman/codes/3_calculate_cer.py --transcript_name {test_name}")
print("CER Calculation Done!\n")

time.sleep(1)
os.system(f"python ./examples/realman/codes/4_summarize_results.py")
print("Results Summarization Done!\n")