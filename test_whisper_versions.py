'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2025-01-23 16:59:15
FilePath: /InASR/test_whisper.py
'''
import whisper

model = whisper.load_model("medium")
result = model.transcribe(
    "/home/fangying/datasets/reverb/et_real_1ch/data/format.1/t21_RealData_et_for_1ch_far_room1_A_t21c0207.flac",
    language="en", temperature=0, beam_size=10, fp16=False)
print(result["text"])

# "We would have to delay any change in production schedules, or else have people sitting around doing nothing, as both of them say."
# torch==2.4.0, openai-whisper==20240930

# "We would have to delay any change in production schedules or else have people sitting around doing nothing, as Bozeman says."
# torch==2.4.0, openai-whisper==20230308

# "We would have to delay any change in production schedules or else have people sitting around doing nothing, as Bozeman says."
# torch==1.12.0, openai-whisper==20230308