import fire
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import copy
import json
from torch.utils.data import Dataset

import pickle as pkl
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import GenerationConfig
import transformers

from llama_recipes.datasets.arc_dataset import ArcDataset
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


class DatasetConfigurations:
    def __init__(self):
        self.seed = 42
        self.work_root = "/oscar/data/csun45/jbyers3/ARC"  # this is a cloned repo from https://github.com/fchollet/ARC
        self.max_tokens = 1000
        self.data_size = -1


def update_dataset_config(config, task, data_type, data_size):
    config.data_size = data_size
    #config.data_root = os.path.join(config.work_root, "data")
    config.data_root = os.path.join(config.work_root, "data/evaluation") #"categorized-data/bin_5")
    return config

def get_data(split, cfg):
    data = []
    """
    if split == "train":
        directory = os.path.join(cfg.data_root, "training")
    elif split == "test":
        directory = os.path.join(cfg.data_root, "evaluation")
    else:
        raise ValueError("Invalid input for the variable split. Must be test or train")
    """
    directory = cfg.data_root

    # option to change number of samples based on cfg.data size variable
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]

    for filename in json_files:  # open every file in the "training" folder
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_content = json.load(file)
                data.append(json_content)

    return data

    
# this function produces the prompts as strings as well as the expected string outputs 
def get_prompts_and_outputs(dataset, dataset_config, is_inference):
    prompts, outputs = [],[]

    for ex_id, ex in tqdm(enumerate(dataset), total=len(dataset)):

        base_prompt = "You will be given a list of input-output pairs, labeled 'Input', 'Output', and so on. "
        base_prompt += "Each input and output is a grid of numbers representing a visual grid. " 
        base_prompt += "There is a SINGLE rule that transforms each input grid to the corresponding output grid. " 
        base_prompt += "Generate the Output grid that corresponds to the last given Input grid, using the transformation rule you induced from the previous input-output pairs."   
        
        examples = ex["train"]
        for example in examples:
            base_prompt += "\nInput: "
            base_prompt += f"{example['input']}\n"
            base_prompt += "Output: "
            base_prompt += f"{example['output']}"

        question = ex["test"][0]
        base_prompt += "\nInput:"
        base_prompt += f"{question['input']}\n"
        base_prompt += "Output:"

        correct_output = f"{question['output']}"

        #base_prompt = "to make a pizza start by"
        prompts.append(base_prompt)
        outputs.append(correct_output)

    return prompts, outputs


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens: int=1, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool=False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    task: str=None, # Task name of ACRE dataset: ["acre", "acre_consistent"]
    data_type: str=None, # Data type of ACRE dataset: ["symbolic", "language"]
    data_size: int=-1, # Number of data for inference. -1 means all the data.
    output_dir: str=None,
    **kwargs
):

    # Prepare dataset configurations
    dataset_config = DatasetConfigurations()
    dataset_config = update_dataset_config(dataset_config, task, data_type, data_size)

    # Prepare prompts
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})


    dataset_config = dataset_config
    split="train"
    is_inference=True
    dataset = get_data(split, dataset_config)

    prompts, outputs = get_prompts_and_outputs(dataset, dataset_config, is_inference)

    shapes = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        example = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        shapes.append(example.shape[0])
    
    print("shapes: " + str(shapes))
        
    x_values = shapes
    bin_edges = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000]
    num_bins = len(bin_edges) - 1

    binned_y_sums = [0] * num_bins
    for x in x_values:
        for i in range(num_bins):
            if bin_edges[i] <= x < bin_edges[i + 1]:
                binned_y_sums[i] += 1
    

    # Calculate the average x value for each bin
    #x_bin_centers = [np.mean(bin) for bin in zip(bin_edges[:-1], bin_edges[1:])]
    x_bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]

    # Plot the data
    plt.bar(x_bin_centers, binned_y_sums, align='center')
    plt.xlabel('Token Number')
    plt.ylabel('Count')
    plt.title('Frequency Token Counts')
    plt.xticks(bin_edges)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    plt.show()
    plt.savefig('total-tokens.png')

if __name__ == "__main__":
    fire.Fire(main)

