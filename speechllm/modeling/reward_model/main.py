import logging
import math

import torch
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

from speechllm.modeling.reward_model.config import DataTrainingArguments, ModelArguments
from speechllm.modeling.reward_model.data_utils.data_collator import (
    DataCollatorforRankingDataset,
)
from speechllm.modeling.reward_model.data_utils.dataloader import RankedDataset
from speechllm.modeling.reward_model.data_utils.utils import Split
from speechllm.modeling.reward_model.metrics import RewardMetrics
from speechllm.modeling.reward_model.reward_model import GPTNeoXRewardModel, GPTJRewardModel, GPTJRewardModelConfig
from speechllm.modeling.reward_model.trainer import RankedTrainer

from loguru import logger

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    (  # pylint: disable=unbalanced-tuple-unpacking
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO,
    # )
    # Log on each process the small summary:
    # logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    log_level = training_args.get_process_log_level()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )


    # model = GPTNeoXRewardModel.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     torch_dtype=torch_dtype,
    # )

    model = GPTJRewardModel(config)
    model.gpt_j = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
    )

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     torch_dtype=torch_dtype,
    # )

    if model.config.pad_token_id is None: 
        model.config.pad_token_id = model.config.eos_token_id

    # model.resize_token_embeddings(len(tokenizer))

    train_dataset = None
    if training_args.do_train:
        train_dataset = RankedDataset(
            data_dir=data_args.data_dir,
            data_name=data_args.data_name,
            tokenizer=tokenizer,
            mode=Split.TRAIN,
            max_sequence_length=data_args.max_sequence_length,
            overwrite_cache=data_args.overwrite_cache,
            augmented=data_args.augmented,
        )

    dev_dataset = None
    if training_args.do_eval:
        dev_dataset = RankedDataset(
            data_dir=data_args.data_dir,
            data_name=data_args.data_name,
            tokenizer=tokenizer,
            mode=Split.VAL,
            max_sequence_length=data_args.max_sequence_length,
            overwrite_cache=data_args.overwrite_cache,
        )

    #print length of train and dev test 
    logger.info(f"train_dataset length: {len(train_dataset)}")
    logger.info(f"dev_dataset length: {len(dev_dataset)}")

    data_collator = DataCollatorforRankingDataset(tokenizer=tokenizer)

    # if data_args.flash_attention:
    #     training_args.remove_unused_columns = False
    #     training_args.torch_compile = True

    compute_metrics = RewardMetrics(["accuracy", "kendalltau", "spearmanr"])

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=4
    )

    # Initialize our Trainer
    trainer = RankedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):  # pylint: disable=unused-argument
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
