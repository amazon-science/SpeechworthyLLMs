# Speechworthy Instruction-tuned Language Models 

Official Repository for [Speechworthy Instruction-tuned Language Models](https://aclanthology.org/2024.emnlp-main.595/) by Cho et. al, presented at EMNLP 2024. 

## Setup 

```bash
conda create -n speechllm python=3.10 
conda activate speechllm
pip install -e . 
pip install -r requirements.txt 
```

## Data 

- Filtered Dolly-15K for voice-suitable prompts: available in `assets/databricks_dolly_voice_prompt_samples_full.jsonl` (TBD: prepare as huggingface dataset)
- SpeechPref Data (TBD: prepare as huggingface dataset)
```
# with wget
wget --recursive --no-parent https://speechllm-emnlp2024.s3.us-west-2.amazonaws.com/llamafactory_data/ -P data/
```


## Model training 

- Follow the steps in our forked repo of [LLaMA Factory](https://github.com/wise-east/llama-factory)
    - Generating response: `data_collection/generate_responses_script_<model>.sh`
- To train a GPT-J-based speech preference reward model that can be used with `speechllm` to compute speechpref reward scores: `speechllm/modeling/reward_model/finetune_reward.sh`, adapted from [trlX](https://github.com/CarperAI/trlx)
    - Download a trained model at `https://speechllm-emnlp2024.s3.us-west-2.amazonaws.com/gptj_reward_model/` (TBD: upload as huggingface model)

```bash 
wget --recursive --no-parent https://speechllm-emnlp2024.s3.us-west-2.amazonaws.com/gptj_reward_model/ -P assets/gptj_reward_model/
```


## Directory guide 

- `speechllm`: main package for speechllm modeling, benchmarking, and utils 
- `data_collection`: scripts for preparing data for preference annotations and their post-processing 
    - `data_collection/hakiki_data_processing`: preparing data for Hakiki 
    - `data_collection/hakiki_templates`: preference annotations and human evaluation templates for Hakiki 
    - Hakiki is an internal annotation framework that's not open source, but the templates are provided as reference. 


## Data processing 

### Preference annotations data preparation 
- Start with [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/viewer/databricks--databricks-dolly-15k/train?row=5) from HuggingFace datasets
- Get voice-suitable prompts by with filters and asking GPT-3.5: `extract_speech_suitable_prompts_from_dolly15k.py` 
- Generate responses from various models: `generate_preference_annotations_data_from_filtered_dolly15k.py`
- Put the responses together into a single file: `combine_response_files.py` (also adds audio for responses and instructions that didn't have audio created)
- Create data for hakiki preference annotations: `hakiki_data_processing/create_preference_annotation_data.py`

### Post model training evaluation 
- Generate responses to use for human evaluation: `data_collection/generate_responses_for_human_evaluation.py`
- Compile human evaluation results or compare preference annotation ranking: `data_collection/hakiki_data_processing/compare_llm_rank_from_annotations.py`
- Get automatic evaluation results for generated responses: `speechllm/evaluate_test_results.py`


## Test

Run tests with: 
```bash
pytest
```

## Citation

If you use this code or data, please cite our paper: 

```
@inproceedings{cho-etal-2024-speechworthy,
    title = "Speechworthy Instruction-tuned Language Models",
    author = "Cho, Hyundong Justin  and
      Jedema, Nicolaas Paul  and
      Ribeiro, Leonardo F. R.  and
      Sharma, Karishma  and
      Szekely, Pedro  and
      Moschitti, Alessandro  and
      Janssen, Ruben  and
      May, Jonathan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.595/",
    doi = "10.18653/v1/2024.emnlp-main.595",
    pages = "10652--10670",
    abstract = "Current instruction-tuned language models are exclusively trained with textual preference data and thus may not be aligned to the unique requirements of other modalities, such as speech. To better align language models with the speech domain, we explore i) prompting strategies based on radio-industry best practices and ii) preference learning using a novel speech-based preference data of 20K samples collected by annotators who listen to response pairs. Both human and automatic evaluation show that both prompting and preference learning increase the speech-suitability of popular instruction tuned LLMs. More interestingly, we show that these methods are additive; combining them achieves the best win rates in head-to-head comparison, resulting in responses that are preferred or tied to the base model in 76.2{\%} of comparisons on average. Lastly, we share lexical, syntactical, and qualitative analyses that elicit how our studied methods differ with baselines in generating more speech-suitable responses."
} 
```


## License

Code is released under CC-by-NC 4.0; Data is released under CC-by-SA. See individual LICENSE.txt in each repo.

