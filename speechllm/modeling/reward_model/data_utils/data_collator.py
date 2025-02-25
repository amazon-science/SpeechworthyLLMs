from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch


@dataclass
class DataCollatorforRankingDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        target_lens = [0]
        n_samples = 0
        for example in instances:
            n_samples += example["input_ids"].shape[0]
            target_lens.append(n_samples)

        input_ids = torch.cat([f["input_ids"] for f in instances], dim=0)
        attention_mask = torch.cat([f["attention_mask"] for f in instances], dim=0)

        return (
            dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
            target_lens,
        )
