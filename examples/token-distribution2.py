import fire
import os
import sys
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import shutil

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
        self.max_tokens = 4096
        self.data_size = -1


def update_dataset_config(config, task, data_type, data_size):
    config.data_size = data_size
    #config.data_root = os.path.join(config.work_root, "data")
    config.data_root = os.path.join(config.work_root, "data/evaluation") #"categorized-data/bin_5")
    return config


def split_files(tokenizer, max_tokens, cfg):
    
    directory = cfg.data_root
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]

    for filename in json_files:  # open every file in the "training" folder
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                json_content = json.load(file)
                ex = json_content

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

                prompt = base_prompt
                example_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
                solution_tokens = torch.tensor(tokenizer.encode(correct_output), dtype=torch.int64)

                total_tokens = example_tokens.shape[0] + solution_tokens.shape[0]

                small_buffer = 5 # small buffer to not exceed max_tokens  
                if total_tokens + small_buffer < max_tokens:  # if the number of tokens is less than 4096
                    base_dest_dir = "/oscar/data/csun45/jbyers3/ARC/categorized-data"
                    dest_dir = base_dest_dir + f"/max_tokens_{max_tokens}"
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(filepath, dest_dir)

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

    # split the files in the ARC directory
    split_files(tokenizer, dataset_config.max_tokens, dataset_config)
    
        
    

if __name__ == "__main__":
    fire.Fire(main)

