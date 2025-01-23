'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2025-01-23 16:59:15
FilePath: /InASR/test_whisper.py
'''
import whisper

model = whisper.load_model("medium")
result = model.transcribe(
    "/data/home/fangying/espnet/egs2/reverb/asr1/dump_16/raw/et_real_1ch/data/format.1/t21_RealData_et_for_1ch_far_room1_A_t21c0207.flac",
    language="en", temperature=0, beam_size=10, fp16=False)
print(result["text"])