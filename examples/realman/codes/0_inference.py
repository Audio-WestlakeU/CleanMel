'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2025-03-05 16:45:09
FilePath: /InASR/examples/realman/codes/0_inference.py
'''

import os
import numpy as np
import argparse

os.environ["OMP_NUM_THREADS"] = str(2)  
# limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine 
lang = 'zh'
fs = 16000  

import time
# import torch

import resampy
# import pandas as pd
import soundfile

import json
import tqdm
from typing import List
import multiprocessing as mp
import numpy as np
import argparse
import string
import logging
import sys

parser = argparse.ArgumentParser(description='Test path and Output path.')
parser.add_argument('--json_file', type=str, default=os.path.join("./examples/realman/results", '4xSPB_Hid96_fixnorm_offline_ensemble89-99.json'), help='Output path.')
args = parser.parse_args()

test_tar = "/data/home/RealisticAudio/RealMAN_modified/test/ma_noisy_speech"

logging.basicConfig(
    filename=f'/data/home/fangying/InASR/examples/realman/results/logs/{args.json_file.split("/")[-1].split(".")[0]}.log',  # 指定日志文件名
    filemode='a',        
    format='%(asctime)s - %(levelname)s - %(message)s',  
    level=logging.INFO   
)


def process(files):
    
    def text_normalizer(text):
        text = text.upper()
        return text.translate(str.maketrans('', '', string.punctuation))

    try:
        sys.path.append('/data/home/fangying/InASR')
        from inference_enh import Speech2Text
        import string
        speech2text = Speech2Text(
            asr_train_config='./examples/realman/config.yaml',
            asr_model_file='./examples/realman/valid.acc.ave_10best.pth',
            device="cpu",
            minlenratio=0.0,
            maxlenratio=0.0,
            ctc_weight=0.6,
            beam_size=10,
            batch_size=0,
            nbest=1
        )
        for index, file in enumerate(files):            
            wav_path = file
            speech, rate = soundfile.read(wav_path)
            speech = resampy.resample(speech, rate, fs, axis=0)
            name = file.split('/')[-1].split('.')[0]
            nbests = speech2text(name, speech)
            text, *_ = nbests[0]
            json_obj = json.dumps([wav_path, text_normalizer(text)], ensure_ascii=False)
            excep=True
            while excep:
                try :
                    file_output = open(args.json_file, 'a', encoding='utf-8')
                    excep=False
                except:
                    time.sleep(1)
                
            file_output.write(json_obj + ',\n')
            file_output.close()

            print(f"{wav_path}: {text_normalizer(text)}")
            print("*" * 50)
    except Exception as e:
        print(f"{e}")

def init_env_var(gpus: List[int]):
    i = gpus.get()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    # print(os.getpid(),os.environ['CUDA_VISIBLE_DEVICES'])


if __name__ == '__main__':
    with open(args.json_file, 'w', encoding='utf-8') as file_output:
        file_output.write('[\n')

    # gpus = list([0,1,2,3,4,7])
    gpus = list(range(8)) # number of jobs
    
    mp.set_start_method('spawn')

    queue = mp.Queue()
    for gid in gpus:
        queue.put(gid)

    all_files = []
    for root, dirs, files in os.walk(test_tar):
        for index, file in enumerate(files):
            if file.endswith('CH0.flac') and '_S_' in file:
                all_files.append(os.path.join(root, file))

    files = all_files
    print(len(files))
    files.sort()

    pbar = tqdm.tqdm(total=len(files))
    
    p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))    

    filess = np.array_split(files, len(gpus))

    p.map(process, filess)
    p.close()
    p.join()

    with open(args.json_file, 'r', encoding='utf-8') as  file_output:
        content = file_output.read()
 
    last_comma_index = content.rfind(',')
    if last_comma_index != -1:
        content = content[:last_comma_index] + content[last_comma_index + 1:] + ']'

    with open(args.json_file, 'w', encoding='utf-8') as  file_output:
        file_output.write(content)

