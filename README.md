# ASR Inference for CleanMel
This branch provides code for performing ASR testing on enhanced mel (.npy format) using pre-trained models from [ESPnet](https://github.com/espnet/espnet) toolkit. Using this branch does not require the installation of ESPnet.

If you are using models trained with other ASR toolkits (e.g. FunASR, WeNet, Whisper, etc.), you can also refer to the implementation in this branch. It is important to note that different ASR toolkits have varying default settings for STFT, the base of the logarithm used for mel, and the methods for normalizing mel.

## Files Introduction
- **examples**:
- **enhasr_test.sh**:
- **inference_enh.py**:
- **networks**:
- **utils**:

## Reproduction Process
### Step 1: Installation
```
pip install -r requirements.txt
```

### Step 2: Prepare data

### Step 3: Prepare models

### Step 4: Inference

## Pretrained ASR Models
Below is a detailed introduction to the specific models tested on different datasets:
1. **REVERB**: Transformer, trained on the REVERB dataset. Pretrained model: [Download Link](https://zenodo.org/record/4441309/files/asr_train_asr_transformer2_raw_en_char_rir_scpdatareverb_rir_singlewav.scp_noise_db_range12_17_noise_scpdatareverb_noise_singlewav.scp_speech_volume_normalize1.0_num_workers2_rir_apply_prob0.999_noise_apply_prob1._sp_valid.acc.ave.zip?download=1)
2. **RealMAN**: Conformer, trained on the WenetSpeech dataset. Pretrained model: [Download Link](https://huggingface.co/espnet/pengcheng_guo_wenetspeech_asr_train_asr_raw_zh_char)
3. **CHiME4**: E-Branchformer, trained on the CHiME4 dataset. Due to the pretrained model for this dataset in ESPnet using an STFT setting with a hop size of 10ms, we modified the STFT settings in [this model](https://huggingface.co/pyf98/chime4_e_branchformer_e10) and retrained the ASR backend to maintain consistency with other models. 