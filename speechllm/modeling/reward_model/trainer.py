from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer


class RankedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_logits=False):
        batch, cu_lengths = inputs
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        losses = []
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)
            l2 = 0.5 * (pos_logits**2 + neg_logits**2)

            beta = 0.001

            _loss = (-F.logsigmoid(pos_logits - neg_logits) + beta * l2).mean()
            losses.append(_loss)

        loss = torch.stack(losses)
        loss = loss.mean()

        return (loss, logits) if return_logits else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], List[int]],
        prediction_loss_only: bool,  # pylint: disable=unused-argument
        ignore_keys: Optional[List[str]] = None,  # pylint: disable=unused-argument
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, cu_lens = inputs
        with torch.no_grad():
            batch = self._prepare_inputs(batch)
            loss, logits = self.compute_loss(model, (batch, cu_lens), return_logits=True)

        loss = loss.mean().detach()

        labels = []
        for i, (s, e) in enumerate(zip(cu_lens[:-1], cu_lens[1:])):
            labels.extend([i] * (e - s))

        # make sure labels are same as logits, needed for deepspeed
        labels = torch.tensor(labels, device=logits.device, requires_grad=False).view(-1, 1)
        return (loss, logits.T, labels.T)  # transposed to avoid truncation in evaluation_loop
