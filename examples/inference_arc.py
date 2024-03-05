# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import pickle as pkl
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import GenerationConfig

from llama_recipes.datasets.arc_dataset import ArcDataset
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


NUM_EXAMPLES_DICT = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
}

class DatasetConfigurations:
    def __init__(self):
        self.seed = 42
        self.work_root = "/oscar/data/csun45/jbyers3/ARC"  # this is a cloned repo from https://github.com/fchollet/ARC
        self.max_tokens = 700
        self.data_size = -1


def update_dataset_config(config, task, data_type, data_size):
    config.data_size = data_size
    config.data_root = os.path.join(config.work_root, "data")
    return config


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
    dataset_test = ArcDataset(dataset_config, tokenizer, split="test", is_inference=True)
    print(f"dataset size: {len(dataset_test)}", flush=True)
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    if peft_model and peft_model != "skip":
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    # model.resize_token_embeddings(model.config.vocab_size + 1)  # for the added padding token
    
    
    responses = []
    for ex_id, batch in tqdm(enumerate(dataset_test)):


        batch = {k: v.unsqueeze(0).to("cuda") for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(input_ids=batch["input_ids"])

        
        first_elements = outputs.logits.argsort(dim=-1)[0][:, -1]
        first_row = outputs.logits.argsort(dim=-1)[0][0]

        #output_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0][-1], skip_special_tokens=True)  # original line
        output_text = tokenizer.decode(first_elements, skip_special_tokens=True)
        #print(output_text)  # allows you to see output text
        
        responses.append(output_text)
        

    # Save responses
    print("-"*50)
    print("Collected responses done!")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "responses.pkl")
    with open(output_path, "wb") as f:
        pkl.dump(responses, f)
    print(f"responses save to {output_path}")
    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)