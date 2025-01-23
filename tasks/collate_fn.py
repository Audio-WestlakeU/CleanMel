import math
from typing import Collection, Dict, List, Tuple, Union

import numpy as np
import torch
# from typeguard import typechecked
from utils.nets_utils import pad_list

class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        # assert typechecked()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


class HuBERTCollateFn(CommonCollateFn):
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        label_downsampling: int = 1,
        pad: bool = False,
        rand_crop: bool = True,
        crop_audio: bool = True,
        not_sequence: Collection[str] = (),
    ):
        # assert typechecked()
        super().__init__(
            float_pad_value=float_pad_value,
            int_pad_value=int_pad_value,
            not_sequence=not_sequence,
        )
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.label_downsampling = label_downsampling
        self.pad = pad
        self.rand_crop = rand_crop
        self.crop_audio = crop_audio
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value}, "
            f"label_downsampling={self.label_downsampling}, "
            f"pad_value={self.pad}, rand_crop={self.rand_crop}) "
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        assert "speech" in data[0][1]
        assert "text" in data[0][1]
        if self.pad:
            num_frames = max([sample["speech"].shape[0] for uid, sample in data])
        else:
            num_frames = min([sample["speech"].shape[0] for uid, sample in data])

        new_data = []
        for uid, sample in data:
            waveform, label = sample["speech"], sample["text"]
            assert waveform.ndim == 1
            length = waveform.size
            # The MFCC feature is 10ms per frame, while the HuBERT's transformer output
            # is 20ms per frame. Downsample the KMeans label if it's generated by MFCC
            # features.
            if self.label_downsampling > 1:
                label = label[:: self.label_downsampling]
            if self.crop_audio:
                waveform, label, length = _crop_audio_label(
                    waveform, label, length, num_frames, self.rand_crop
                )
            new_data.append((uid, dict(speech=waveform, text=label)))

        return common_collate_fn(
            new_data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def _crop_audio_label(
    waveform: torch.Tensor,
    label: torch.Tensor,
    length: torch.Tensor,
    num_frames: int,
    rand_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate the audio and label at the same time.

    Args:
        waveform (Tensor): The waveform Tensor with dimensions `(time)`.
        label (Tensor): The label Tensor with dimensions `(seq)`.
        length (Tensor): The length Tensor with dimension `(1,)`.
        num_frames (int): The final length of the waveform.
        rand_crop (bool): if ``rand_crop`` is True, the starting index of the
            waveform and label is random if the length is longer than the minimum
            length in the mini-batch.

    Returns:
        (Tuple(Tensor, Tensor, Tensor)): Returns the Tensors for the waveform,
            label, and the waveform length.

    """

    kernel_size = 25
    stride = 20
    sample_rate = 16  # 16 per millisecond
    frame_offset = 0
    if waveform.size > num_frames and rand_crop:
        diff = waveform.size - num_frames
        frame_offset = torch.randint(diff, size=(1,))
    elif waveform.size < num_frames:
        num_frames = waveform.size
    label_offset = max(
        math.floor((frame_offset - kernel_size * sample_rate) / (stride * sample_rate))
        + 1,
        0,
    )
    num_label = (
        math.floor((num_frames - kernel_size * sample_rate) / (stride * sample_rate))
        + 1
    )
    waveform = waveform[frame_offset : frame_offset + num_frames]
    label = label[label_offset : label_offset + num_label]
    length = num_frames

    return waveform, label, length


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    # assert typechecked()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    # assert typechecked(output)
    return output
