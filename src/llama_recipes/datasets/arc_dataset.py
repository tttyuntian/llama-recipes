# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://github.com/fchollet/ARC

import copy
import json
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# returns all the json data from the train or test split
def get_data(split, cfg):
    data = []
    if split == "train":
        directory = os.path.join(cfg.data_root, "training")
    elif split == "test":
        directory = os.path.join(cfg.data_root, "evaluation")
    else:
        raise ValueError("Invalid input for the variable split. Must be test or train")

    # option to change number of samples based on cfg.data size variable
    json_files = [filename for filename in os.listdir(directory) if filename.endswith('.json')]
    if cfg.data_size == -1:
        data_size = len(json_files)
    else:
        data_size = min(cfg.data_size, len(json_files))

    for filename in json_files[:data_size]:  # open every file in the "training" folder
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
            base_prompt += "\nInput: \n"
            base_prompt += f"{example['input']}\n"
            base_prompt += "Output: \n"
            base_prompt += f"{example['output']}"

        question = ex["test"][0]
        base_prompt += "\nInput: \n"
        base_prompt += f"{question['input']}\n"

        correct_output = f"{question['output']}"


        prompts.append(base_prompt)
        outputs.append(correct_output)


    return prompts, outputs


class ArcDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train", is_inference=False):
        self.dataset_config = dataset_config
        self.dataset = get_data(split, dataset_config)

        self.prompts, self.outputs = get_prompts_and_outputs(self.dataset, dataset_config, is_inference)

        self.tokenizer = tokenizer
        self.max_tokens = dataset_config.max_tokens
        self.is_inference = is_inference


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        if not self.is_inference:
            prompt, output = self.prompts[idx], self.outputs[idx]
            example = prompt + output
            prompt = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(
                example, dtype=torch.int64
            )
            
            padding = self.max_tokens - example.shape[0]
            if padding > 0:
                input_ids = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_ids = example[: self.max_tokens]
                print(f"Example {idx} with {example.shape[0]} tokens goes over max_tokens={self.max_tokens}")
            
            labels = copy.deepcopy(input_ids)
            labels[: len(prompt)] = -1
            input_ids_mask = input_ids.ge(0)
            label_mask = labels.ge(0)
            input_ids[~input_ids_mask] = 0
            labels[~label_mask] = IGNORE_INDEX
            input_ids_mask = input_ids_mask.float()
            label_mask = label_mask.float()
        
        else:
            # Assume batch_size=1 during model inference
            prompt, output = self.prompts[idx], self.outputs[idx]
            example = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            labels = torch.tensor(self.tokenizer.encode(output, add_special_tokens=False), dtype=torch.int64)
            padding = self.max_tokens - example.shape[0]
            if padding > 0:
                input_ids = example[:]
            elif padding < 0:
                input_ids = example[: self.max_tokens]
                print(f"Example {idx} with {example.shape[0]} tokens goes over max_tokens={self.max_tokens}")
            elif padding == 0:
                input_ids = example[:]
                
            input_ids_mask = input_ids.ge(0)
            input_ids[~input_ids_mask] = 0
            input_ids_mask = input_ids_mask.float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids_mask,
        }
