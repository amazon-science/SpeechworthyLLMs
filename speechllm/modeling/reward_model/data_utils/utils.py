from dataclasses import dataclass
from enum import Enum
from typing import List

import torch


@dataclass(frozen=True)
class InputExample:
    """A single training/test example.

    Args:
        target_text: string. The untokenized text of the target.
    """

    target_texts: List[str]


@dataclass(frozen=True)
class InputFeatures:
    """A single set of features of data.

    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
