'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-11-25 16:15:01
FilePath: /InASR/examples/wenetspeech/data/Anker/organize.py
'''
import os

flac_path = "/data/home/fangying/sn_enh_mel/Anker_new_transcript/SmallModel/5m_chunks/unproc"
scp_path = "/data/home/fangying/InASR/examples/wenetspeech/data/anker_5m/wav.scp"

if os.path.exists(scp_path):
    os.remove(scp_path)

for root, dirs, files in os.walk(flac_path):
    for file in files:
        if file.endswith(".flac"):
            file_path = os.path.join(root, file)
            file_name = file.split(".")[0]
            with open(scp_path, "a") as f:
                f.write(file_name + " " + file_path + "\n")