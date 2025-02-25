# use generate_responses_for_human_evaluation_updated.py to generate responses for human evaluation from various models 
PACKAGE_DIR=$(dirname $(dirname $(realpath $0)))

conda activate speechllm

export CUDA_VISIBLE_DEVICES=2

OLMO_BASE="allenai/OLMo-7B-Instruct-hf"
SAVE_PATH=${PACKAGE_DIR}/assets/olmo_human_eval_responses.jsonl
# TEST="--test"
TEST=""

# absolute base
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path=$OLMO_BASE \
    --not_lora \
    --system_prompt_type base \
    --save_path $SAVE_PATH \
    $TEST

# with detailed prompt
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path=$OLMO_BASE \
    --not_lora \
    --system_prompt_type detailed \
    --save_path $SAVE_PATH \
    $TEST

# with icl 
python generate_responses_for_human_evaluation_updated.py \
    --model_name_or_path=$OLMO_BASE \
    --not_lora \
    --system_prompt_type icl \
    --save_path $SAVE_PATH \
    $TEST

# PPO models
PPO_BASE_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/ppo-base
PPO_DETAILED_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/ppo-detailed/
PPO_ICL_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/ppo-icl/
# DPO models 
DPO_BASE_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/dpo-base
DPO_DETAILED_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/dpo-detailed
DPO_ICL_PATH=${HOME}/project/LLaMA-Factory/saves/olmo/dpo-icl

PATH_LISTS=($PPO_BASE_PATH $PPO_DETAILED_PATH $PPO_ICL_PATH $DPO_BASE_PATH $DPO_DETAILED_PATH $DPO_ICL_PATH)
for RLHF_PATH in ${PATH_LISTS[@]};
do
    echo "Generating responses for $RLHF_PATH"
    python generate_responses_for_human_evaluation_updated.py \
        --model_name_or_path=$RLHF_PATH \
        --system_prompt_type base \
        --save_path $SAVE_PATH \
        $TEST

done

PATH_LISTS=($PPO_DETAILED_PATH $DPO_DETAILED_PATH)
for RLHF_PATH in ${PATH_LISTS[@]};
do
    python generate_responses_for_human_evaluation_updated.py \
        --model_name_or_path=$RLHF_PATH \
        --system_prompt_type detailed \
        --save_path $SAVE_PATH \
        $TEST

done 

PATH_LISTS=($PPO_ICL_PATH $DPO_ICL_PATH)
for RLHF_PATH in ${PATH_LISTS[@]};
do
    python generate_responses_for_human_evaluation_updated.py \
        --model_name_or_path=$RLHF_PATH \
        --system_prompt_type icl \
        --save_path $SAVE_PATH \
        $TEST
done


