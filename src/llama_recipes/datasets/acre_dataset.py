# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import json
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


NUM_EXAMPLES_DICT = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
}


IDX_TO_LIGHT_STATE = {
    0: "off",
    1: "unknown",
    2: "on",
}


def get_data_scenes(split, cfg, num_examples, num_panels_per_example=10):
    
    def parse_context(context):
        return {
            "objects": [
                f"{obj['color']} {obj['material']} {obj['shape']}"
                for obj in context["objects"]
            ],
            "light_state": context["light_state"],
        }

    with open(os.path.join(cfg.data_split_root, "config", f"{split}.json"), "r") as f:
        data_config = json.load(f)

    data_scenes = []
    for i in tqdm(range(num_examples), desc=f"{split}"):
        if i == cfg.data_size:
            break
        ex = []
        for k in range(num_panels_per_example):
            file_path = os.path.join(cfg.data_split_root, "scenes", f"ACRE_{split}_00{i:04}_{k:02}.json")
            with open(file_path, "rb") as f:
                context = json.load(f)
            parsed_context = parse_context(context)

            if k in range(cfg.num_contexts_per_example, num_panels_per_example):
                parsed_context["label"] = data_config[i][k]["label"]
                parsed_context["type"] = data_config[i][k]["type"]

            ex.append(parsed_context)

        data_scenes.append(ex.copy())
    return data_scenes


def get_data(data_type, split, cfg):
    num_examples = NUM_EXAMPLES_DICT[split]
    if cfg.task == "acre":
        if data_type == "symbolic":
            with open(os.path.join(cfg.data_split_root, "config", f"{split}.json"), "r") as f:
                data = json.load(f)
        elif data_type == "language":
            data = get_data_scenes(split, cfg, num_examples, cfg.num_panels_per_example)
    elif cfg.task in ["acre_indirect_two_objects", "acre_consistent"]:
        if data_type == "symbolic":
            with open(os.path.join(cfg.data_split_root, "config", f"{split}.json"), "r") as f:
                data = json.load(f)
        elif data_type == "language":
            with open(os.path.join(cfg.data_split_root, "language", f"{split}.json"), "r") as f:
                data = json.load(f)
    return data


def get_objects_str(objects, data_type):
    if data_type == "symbolic":
        return f"Objects: {objects}"
    elif data_type == "language":
        num_objects = len(objects)
        if num_objects == 1:
            return f"Objects: There is {objects[0]}."
        elif num_objects == 2:
            return f"Objects: There are {objects[0]} and {objects[1]}."
        else:
            return f"Objects: There are {', '.join(objects[:-1])} and {objects[-1]}."
    else:
        raise Exception(f"{data_type} not supported.")
    

def get_prompts_and_outputs(dataset, dataset_config, is_inference):
    prompts, outputs = [],[]

    for ex_id, ex in tqdm(enumerate(dataset), total=len(dataset)):

        base_prompt = "You are a helpful assistant that determines whether the light will be activated by the objects."
        if dataset_config.data_type == "symbolic":
            base_prompt = "Each object is represented by an integer."
        elif dataset_config.data_type == "language":
            base_prompt = "Each object is described by a color, texture, and shape."
        base_prompt += "Some objects can activate the light. The other objects cannot activate the light. There are three possible light states: on, off, and unknown.\n"

        for i in range(dataset_config.num_contexts_per_example):
            panel = ex[i]
            light_state = "unknown" if panel["light_state"] == "undetermined" else panel["light_state"]
            base_prompt += f"{get_objects_str(panel['objects'], dataset_config.data_type)}\nLight: {light_state}\n"

        for i in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            
            panel = ex[i]
            
            is_valid = False
            if not is_inference:
                if dataset_config.train_query_type == "":
                    # No specified train_query_type, then we use all the data for training
                    is_valid = True
                else:
                    if panel["type"] == dataset_config.train_query_type:
                        # If this is train_query_type, only count the target query type
                        is_valid = True
            else: 
                # Run through all inference examples, regardless of query types
                is_valid = True
                # if dataset_config.train_query_type == "":
                #     # No specified train_query_types, then we use all the data for validation/test
                #     is_valid = True
                # else:
                #     if panel["type"] != dataset_config.train_query_type:
                #         # If this is train_query_type, only count the query types which are not the train_query_type
                #         is_valid = True

            if is_valid:
                light_state = IDX_TO_LIGHT_STATE[panel["label"]]
                prompt = base_prompt + f"{get_objects_str(panel['objects'], dataset_config.data_type)}\nLight:"
                prompts.append(prompt)
                outputs.append(light_state)
    return prompts, outputs


class AcreDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, split="train", is_inference=False):
        self.dataset_config = dataset_config
        self.dataset = get_data(dataset_config.data_type, split, dataset_config)
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
