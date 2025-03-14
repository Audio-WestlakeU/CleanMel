import copy
from typing import Optional, Tuple, Union
import logging
import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
# from typeguard import typechecked
import os
from networks.frontend.abs_frontend import AbsFrontend
from networks.layers.log_mel import LogMel
from networks.layers.stft import Stft
from utils.get_default_kwargs import get_default_kwargs
from networks.frontend.frontend import Frontend

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return

class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        # assert typechecked()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        # logging.info("hop_length: " + str(hop_length))

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
            
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

        self.base_mels_path = "/data/home/fangying/sn_enh_mel/mels/8xCleanMel_dp_L1_weightavg89-99"


    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, name, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement

        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2
       
        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)
    
        logging.info(str(input_feats.shape))
        logging.info(f'input_feats: {input_feats}')
        logging.info(f'input_feats: {input_feats[0].max()}, {input_feats[0].min()}')

        if 'room' in name:
            if 'REVERB' not in self.base_mels_path:
                self.base_mels_path = os.path.join(self.base_mels_path, 'REVERB')
        elif 'room' not in name and ('REAL' in name or 'SIMU' in name):
            if 'CHIME' not in self.base_mels_path:
                self.base_mels_path = os.path.join(self.base_mels_path, 'CHIME')
        elif 'CH0' in name:
            if 'RealMAN' not in self.base_mels_path:
                self.base_mels_path = os.path.join(self.base_mels_path, 'RealMAN')
 
        logging.info(f"self.base_mels_path: {self.base_mels_path}")
        ######################################## ON CHIME-4
        if 'CHIME' in self.base_mels_path:
            if 'REAL' in name:
                if 'real' not in self.base_mels_path.split('/')[-1]:
                    self.base_mels_path = os.path.join(self.base_mels_path, 'real')
                name = name[2:-7]
                logging.info("name: " + str(name))
                mel = np.load(os.path.join(self.base_mels_path, name + '_REAL.npy'))
            elif 'SIMU' in name:
                if 'simu' not in self.base_mels_path.split('/')[-1]:
                    self.base_mels_path = os.path.join(self.base_mels_path,'simu')
                name = name[2:-7]
                logging.info("name: " + str(name))
                mel = np.load(os.path.join(self.base_mels_path, name + '_SIMU.npy'))
            else:
                raise ValueError("Unknown dataset")
            
            mel_tensor = torch.from_numpy(mel).to(input_feats.device).reshape(self.n_mels,1,-1).permute(1,2,0)

        ######################################## ON REVERB
        if 'REVERB' in self.base_mels_path:
            if 'Real' in name and 'real' not in self.base_mels_path.split('/')[-1]:
                self.base_mels_path = os.path.join(self.base_mels_path, 'real')
            elif 'Sim' in name and 'simu' not in self.base_mels_path.split('/')[-1]:
                self.base_mels_path = os.path.join(self.base_mels_path,'simu')

            name = name[2:-2]
            logging.info("name: " + str(name))
            mel = np.load(os.path.join(self.base_mels_path, name + '.npy'))
            mel_tensor = torch.from_numpy(mel).to(input_feats.device).reshape(self.n_mels,1,-1).permute(1,2,0)
        
        ######################################## ON RealMAN
        if 'RealMAN' in self.base_mels_path:
            logging.info("name: " + str(name))
            mel = np.load(os.path.join(self.base_mels_path, name + '.npy'))
            mel_tensor = torch.from_numpy(mel).to(input_feats.device).permute(0,2,1)[:,:input_feats.shape[1],:]

        
        mean = mel_tensor.mean(dim=1, keepdim=True)
        logging.info("Mean mel_tensor: " + str(mean))
        
        logging.info(f'mel_tensor: {mel_tensor.shape}')
        logging.info(f'max: {mel_tensor.max()}')
        logging.info(f'min: {mel_tensor.min()}')
        feats_lens = torch.tensor([mel_tensor.shape[1]])
        return mel_tensor, feats_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens

