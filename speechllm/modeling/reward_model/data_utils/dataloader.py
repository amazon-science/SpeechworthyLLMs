import logging
import os
from typing import List, Optional, Sequence

from filelock import FileLock
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset

from speechllm.modeling.reward_model.data_utils.utils import (
    InputExample,
    InputFeatures,
    Split,
)
import json

def read_jsonl(input_file):
    with open(input_file, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]

    return lines 

logger = logging.getLogger(__name__)


class RankedDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_name: str,
        tokenizer: PreTrainedTokenizer,
        mode: Split = Split.TRAIN,
        max_sequence_length: int = 32,
        overwrite_cache: bool = True,
        augmented: bool = False,
    ):
        self.data_dir = data_dir
        self.data_name = data_name
        self.mode = mode
        self.features: Sequence[BatchEncoding] = []
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            os.path.dirname(self.data_dir),
            "cached_{}_{}_{}_{}".format(
                self.data_name,
                self.mode.value,
                tokenizer.__class__.__name__,
                str(self.max_sequence_length),
            ),
        )

        # dataset = load_dataset("Anthropic/hh-rlhf")

        lock_path = cached_features_file + ".lock"
        if os.path.exists(cached_features_file) and not overwrite_cache:
            with FileLock(lock_path):
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {os.path.dirname(self.data_name)}")

            if mode == Split.VAL:
                # maybe we can sample some examples for faster training
                examples = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/test.jsonl")
                # examples = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/train.jsonl")[:1000]
                # examples = dataset["test"]
            elif mode == Split.TEST:
                examples = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/test.jsonl")
                # examples = read_jsonl(os.path.join(self.data_dir, self.data_name) + "/train.jsonl")[:1000]
                # examples = dataset["test"]
            else:
                # fp = "/augmented_train.jsonl"
                fp = "/train.jsonl" 
                if augmented: 
                    fp = "/augmented_train.jsonl"
                examples = read_jsonl(os.path.join(self.data_dir, self.data_name) + fp)
                # examples = dataset["train"]

            # TODO: Processor for each dataset.
            # HH
            final_examples = []
            for example in examples:
                # source_text = example["prompt"]

                chosen = example["chosen"]
                rejected = example["rejected"]
                final_examples.append(
                    InputExample(target_texts=[chosen, rejected])
                )

            logger.info("Num examples: %s", len(final_examples))
            self.features = self.convert_examples_to_features(
                final_examples, max_sequence_length, tokenizer
            )
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(self.features, cached_features_file)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        input_ids = self.features[index].input_ids
        attention_mask = self.features[index].attention_mask
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        max_sequence_length: Optional[int],
        tokenizer: PreTrainedTokenizer,
    ):
        features = []
        total, discard = 0, 0

        for example in tqdm(examples):
            total += 1
            skip_flag = False
            input_ids = []
            attention_masks = []

            for response in example.target_texts:
                # text = example.source_text + "<|sep|>" + response + tokenizer.eos_token
                text = response + tokenizer.eos_token
                feature = tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=max_sequence_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )

                if len(feature["overflow_to_sample_mapping"]) > 1:
                    discard += 1
                    skip_flag = True

                input_ids.append(feature["input_ids"].view(1, -1))
                attention_masks.append(feature["attention_mask"].view(1, -1))

            if skip_flag:
                continue

            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                )
            )

        print(f"Total: {total} / Discarded: {discard}")
        return features
