# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
import json
import os
from tqdm import tqdm


NUM_EXAMPLES_DICT = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
}


IDX_TO_LIGHT_STATE = {
    0: "off",
    1: "undetermined",
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


def get_data(data_type, split, cfg, num_panels_per_example=10):
    num_examples = NUM_EXAMPLES_DICT[split]
    if cfg.task == "acre":
        if data_type == "symbolic":
            with open(os.path.join(cfg.data_split_root, "config", f"{split}.json"), "r") as f:
                data = json.load(f)
        elif data_type == "language":
            data = get_data_scenes(split, cfg, num_examples, num_panels_per_example)
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
    

def get_prompts(dataset, dataset_config):
    prompts = []

    for ex_id, ex in tqdm(enumerate(dataset), total=len(dataset)):

        base_prompt = "You are a helpful assistant that determines whether the light will be activated by the objects."
        if dataset_config.data_type == "symbolic":
            base_prompt = "Each object is represented by an integer."
        elif dataset_config.data_type == "language":
            base_prompt = "Each object is described by a color, texture, and shape."
        base_prompt += "Some objects can activate the light. The other objects cannot activate the light. There are three possible light states: on, off, and unknown.\n"

        for i in range(dataset_config.num_contexts_per_example):
            panel = ex[i]
            base_prompt += f"{get_objects_str(panel['objects'], dataset_config.data_type)}\nLight: {panel['light_state']}\n"

        for i in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            panel = ex[i]
            prompt = base_prompt + f"{get_objects_str(panel['objects'], dataset_config.data_type)}\nLight: {IDX_TO_LIGHT_STATE[panel['label']]}"
            prompts.append(prompt)
    return prompts


def get_custom_dataset(dataset_config, tokenizer, split):
    
    dataset = get_data(dataset_config.data_type, split, dataset_config, num_panels_per_example=dataset_config.num_panels_per_example)

    prompts = get_prompts(dataset, dataset_config)
    dataset = datasets.Dataset.from_dict({"text": prompts})

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    )
    return dataset