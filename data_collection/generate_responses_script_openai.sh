# use generate_responses_for_human_evaluation_updated.py to generate responses for human evaluation from various models 
PACKAGE_DIR=$(dirname $(dirname $(realpath $0)))

conda activate speechllm

MODEL="openai"
MODEL_TYPE="gpt-4o"
SAVE_PATH=${PACKAGE_DIR}/assets/gpt_human_eval_responses.jsonl

# absolute base
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path $MODEL \
    --openai_version $MODEL_TYPE \
    --system_prompt_type base \
    --save_path $SAVE_PATH \
    # --test

# with detailed prompt
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path $MODEL \
    --openai_version $MODEL_TYPE \
    --system_prompt_type detailed \
    --save_path $SAVE_PATH \
    # --test

# with icl 
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path $MODEL \
    --openai_version $MODEL_TYPE \
    --system_prompt_type icl \
    --save_path $SAVE_PATH \
    # --test
