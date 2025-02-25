PACKAGE_DIR=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))

USE_TF=0 deepspeed --num_gpus=8 main.py \
--model_name_or_path EleutherAI/gpt-j-6B \
--output_dir ./new_reward_model_augment0 \
--data_dir ${PACKAGE_DIR}/assets \
--data_name speech_preference_data \
--max_sequence_length 1024 \
--do_train \
--do_eval \
--evaluation_strategy steps \
--save_strategy steps \
--eval_steps 10 \
--save_steps 10 \
--num_train_epochs 10 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 8 \
--use_fast_tokenizer True \
--learning_rate 1e-6 \
--weight_decay 0.0 \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--torch_dtype bfloat16 \
--bf16 \
--bf16_full_eval True \
--log_level warning \
--tf32 True \
--gradient_checkpointing \
--flash_attention \
--remove_unused_columns False \
--torch_compile True \
--deepspeed deepspeed_configs/ds_bf16.json \
--save_total_limit 3 \
--metric_for_best_model loss \
--load_best_model_at_end True \
--augmented True \


