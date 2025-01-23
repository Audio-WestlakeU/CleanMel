from abc import ABC, abstractmethod
from typing import Tuple

import torch

from search.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
