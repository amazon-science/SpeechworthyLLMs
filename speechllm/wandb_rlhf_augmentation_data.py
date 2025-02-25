# use wandb to collect response comparison data from two different checkpoints
# idea is that degenerate outputs from trained models are always less preferred than the original model's outputs and outputs with very high mean kl per token are degenerate. 

import wandb
import pandas as pd
from argparse import ArgumentParser
import json
from dataclasses import dataclass
from loguru import logger
from speechllm.utils import format_input_string_for_reward_model, PACKAGE_DIR
from pathlib import Path

# Initialize wandb API
api = wandb.Api()

@dataclass
class WandbArgs: 
    project_name: str
    entity_name: str
    run_id: str
    artifact_name: str
    version: str


# outputs with weird tokens and nonsense 
mistral_degen1 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="o170vsdz",
    artifact_name="run",
    version="28"
)

mistral_orig1 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="o170vsdz",
    artifact_name="run",
    version="0"
)

# outputs with weird tokens and nonsense 
mistral_degen2 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="8sggzt25",
    artifact_name="run",
    version="32"
)

mistral_orig2 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="8sggzt25",
    artifact_name="run",
    version="0"
)


# outputs with high reward but repeated 'youyouyou'
olmo_degen1 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="e70tqaj6",
    artifact_name="run",
    version="40"
)

olmo_orig1 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="e70tqaj6", 
    artifact_name="run",
    version="0"
)

# outputs with weird follow up questions 
olmo_degen2 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="88596tt2",
    artifact_name="run",
    version="40" 
)

olmo_orig2 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="88596tt2",  
    artifact_name="run",
    version="0"
)

# outputs with higher rewards than original but degenerate
olmo_degen3 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="9pni2wu8",
    artifact_name="run",
    version="22" 
)

olmo_orig3 = WandbArgs(
    project_name="trlx",
    entity_name="wise-east",
    run_id="9pni2wu8",
    artifact_name="run",
    version="0"
)



def get_wandb_data(wandb_args):

    # Fetch runs from wandb
    runs = api.runs(f"{wandb_args.entity_name}/{wandb_args.project_name}")

    # Replace with your actual run id and artifact names
    artifact_name = f"{wandb_args.artifact_name}-{wandb_args.run_id}-samples:v{wandb_args.version}"

    # Extract tables from run checkpoints
    for run in runs:
        if run.id == wandb_args.run_id:
            # Fetch table from the first checkpoint (v0)
            artifact = api.artifact(f"{wandb_args.entity_name}/{wandb_args.project_name}/{artifact_name}")
            artifact_dir = artifact.download()
            with open(f"{artifact_dir}/samples.table.json") as f:
                data = json.load(f)
                df = pd.DataFrame(data["data"], columns=data["columns"])

    return df 

def get_comparison_data(preferred_response_df, rejected_response_df): 

    # Join the tables on the 'prompt' column

    # Rename columns to include prefixes
    preferred_response_df = preferred_response_df.add_prefix('preferred_')
    rejected_response_df = rejected_response_df.add_prefix('rejected_')

    # Rename the prompt columns back to 'prompt' for merging
    preferred_response_df.rename(columns={'preferred_prompt': 'prompt'}, inplace=True)
    rejected_response_df.rename(columns={'rejected_prompt': 'prompt'}, inplace=True)

    # reformat prompt column 
    preferred_response_df['prompt'] = preferred_response_df['prompt'].apply(format_input_string_for_reward_model)
    rejected_response_df['prompt'] = rejected_response_df['prompt'].apply(format_input_string_for_reward_model)

    # prepend prompt column to the rejected output column and preferred output column
    preferred_response_df['preferred_output'] = preferred_response_df['prompt'] + preferred_response_df['preferred_output']
    rejected_response_df['rejected_output'] = rejected_response_df['prompt'] + rejected_response_df['rejected_output']

    merged_df = pd.merge(preferred_response_df, rejected_response_df, on='prompt')

    logger.info(f"merged_df length: {len(merged_df)}")

    return merged_df

def get_preconfigured_augmentation_data():

    # inputs
    wandb_run_pairs = [
        # (project_name, entity_name, run_id, artifact_name, preferred_version, rejected_version)
        (mistral_degen1, mistral_orig1), 
        (mistral_degen2, mistral_orig2), 
        (mistral_degen1, mistral_orig2),
        (olmo_degen1, olmo_orig1),
        (olmo_degen2, olmo_orig2),
        (olmo_degen3, olmo_orig3),
        (olmo_degen2, olmo_orig3),
        (olmo_degen3, olmo_orig2)
    ]

    merged_dfs = [] 
    for rejected_run, preferred_run in wandb_run_pairs: 
        preferred_response_df = get_wandb_data(preferred_run)
        rejected_response_df = get_wandb_data(rejected_run)

        merged_df = get_comparison_data(preferred_response_df, rejected_response_df)
        merged_dfs.append(merged_df)
        
    # concatenate all the dataframes
    final_df = pd.concat(merged_dfs)

    # do the formatting to match the jsonl file 
    rename_columns = {
        "prompt": "prompt",
        "preferred_output": "chosen",
        "rejected_output": "rejected",
    }

    final_df.rename(columns=rename_columns, inplace=True)

    # open file to append to 
    original_train_path = f"{PACKAGE_DIR}/assets/speech_preference_data/train.jsonl"
    with open(original_train_path, "r") as f:
        original_train_data = [json.loads(line) for line in f.readlines()]
    

    # print length of df
    logger.info(f"final_df length: {len(final_df)}")

    # save augment data only
    augmented_only_data_path = Path(original_train_path).parent / "augmented_only.jsonl"
    final_df.to_json(augmented_only_data_path, orient="records", lines=True)

    # append the new data to the original data
    original_train_data.extend(final_df.to_dict(orient="records"))

    # save the final data to the file
    augmented_train_path = Path(original_train_path).parent / "augmented_train.jsonl"
    with open(augmented_train_path, "w") as f:
        for line in original_train_data:
            f.write(json.dumps(line) + "\n")


def main():

    parser = ArgumentParser()
    parser.add_argument("--project_name", type=str, default="trlx")
    parser.add_argument("--entity_name", type=str, default="wise-east")
    parser.add_argument("--run_id", type=str, default="9pni2wu8")
    parser.add_argument("--artifact_name", type=str, default="run")
    parser.add_argument("--preferred_version", type=str, default="0")
    parser.add_argument("--rejected_version", type=str, default="40")
    parser.add_argument("--preconfigured", action="store_true")

    args = parser.parse_args()

    if args.preconfigured:
        get_preconfigured_augmentation_data() 
        return 

    wandb_args = WandbArgs(
        project_name=args.project_name,
        entity_name=args.entity_name,
        run_id=args.run_id,
        artifact_name=args.artifact_name,
        version=args.preferred_version
    )

    get_wandb_data(wandb_args)
    
if __name__ == "__main__":
    main()


