# script for collecting voice suitable prompts from Databricks Dolly 15k dataset 
# with a rule-based filerter and openai as an evaluator

from argparse import ArgumentParser
from speechllm.dataset_utils import DollyDataset
from speechllm.utils import PACKAGE_DIR

def main(): 

    parser = ArgumentParser()   
    parser.add_argument("--long_response_threshold", type=int, default=50)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--output_fn", type=str, default=f"{PACKAGE_DIR}/assets/databricks_dolly_voice_prompt_samples_full.jsonl")
    args = parser.parse_args()

    dollydataset = DollyDataset(args)
    dollydataset.count_long_responses()

    if args.collect: 
        dollydataset.collect_voice_prompts()
        dollydataset.print_statistics()


if __name__ == "__main__": 
    main() 

