<!--
 * @Author: FnoY fangying@westlake.edu.cn
 * @LastEditors: FnoY0723 fangying@westlake.edu.cn
 * @LastEditTime: 2025-03-14 12:06:14
 * @FilePath: /InASR/README.md
-->
# ASR Inference for CleanMel
This branch provides code for performing ASR testing on enhanced mel (.npy format) using pre-trained models from [ESPnet](https://github.com/espnet/espnet) toolkit. Using this branch does not require the installation of ESPnet.

If you are using models trained with other ASR toolkits (e.g. FunASR, WeNet, Whisper, etc.), you can also refer to the implementation in this branch. It is important to note that different ASR toolkits have varying default settings for STFT, the base of the logarithm used for mel, and the methods for normalizing mel.

## Files Introduction
- [**examples**](./examples): Inside, there are datasets, models, and scripts needed for inference with different datasets.
  
- [**enhasr_test.sh**](./enhasr_test.sh): A script for running inference on three datasets: REVERB, RealMAN, and CHiME4. The input parameter **'mel_path'** is the path that contains the mel npy files for the three datasets, which includes the folders 'REVERB', 'CHIME', and 'RealMAN'.
- [**inference_enh.py**](./inference_enh.py): The script for performing ASR inference. It passes the sample **name** parameter in Speech2Text.
- [**networks**](./networks):Includes the network structure from ESPnet. The enhanced cleanmel replaces the original logmel in [*/InASR/networks/frontend/default_enh.py*](./networks/frontend/default_enh.py).
- **utils**: If the */InASR/utils/sclite* tool is incompatible, run [*/InASR/utils/install_sctk.sh*](./utils/install_sctk.sh) to reinstall it.

## Reproduction Process
### Step 1: Installation
```
pip install -r requirements.txt
```
### Step 2: Prepare data
For REVERB or CHiME4, modify the data paths in *examples/'dataset'/data/'testset'/wav.scp*.
For RealMAN, change the **'test_tar'** path in line 38 of [*0_inference.py*](./examples/realman/codes/0_inference.py).

### Step 3: Prepare models
Download the pretrained ASR model to the corresponding examples/'dataset'/

### Step 4: Inference
```
# Infer on REVERB, CHiME4, and RealMAN datasets
./examples/reverb/test.sh "${mel_path}"
./examples/chime4/test.sh "${mel_path}"
./examples/realman/codes/test.sh "${mel_path}" "lowsnr"
```

## Pretrained ASR Models
Below is a detailed introduction to the specific models tested on different datasets:
1. **REVERB**: Transformer, trained on the REVERB dataset. Pretrained model: [Download Link](https://zenodo.org/record/4441309/files/asr_train_asr_transformer2_raw_en_char_rir_scpdatareverb_rir_singlewav.scp_noise_db_range12_17_noise_scpdatareverb_noise_singlewav.scp_speech_volume_normalize1.0_num_workers2_rir_apply_prob0.999_noise_apply_prob1._sp_valid.acc.ave.zip?download=1)
2. **RealMAN**: Conformer, trained on the WenetSpeech dataset. Pretrained model: [Download Link](https://huggingface.co/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char)
3. **CHiME4**: E-Branchformer, trained on the CHiME4 dataset. Due to the pretrained model for this dataset in ESPnet using an STFT setting with a hop size of 10ms, we modified the STFT settings in [this model](https://huggingface.co/pyf98/chime4_e_branchformer_e10) and retrained the ASR backend to maintain consistency with other models. 