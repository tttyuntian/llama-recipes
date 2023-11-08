# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class acre_dataset:
    dataset: str = "acre_dataset"
    # file: str = "examples/acre_dataset.py"
    train_split: str = "train"
    test_split: str = "val"
    task: str = "acre"
    data_split: str = "iid"
    data_size: int = -1
    max_tokens: int = 312
    data_split_root: str = f"/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning/data/{task}/{data_split}"
    data_type: str = "language"  # ["symbolic", "language"]
    num_contexts_per_example = 6
    num_queries_per_example = 4
    num_panels_per_example = 10
    train_query_type: str = ""


@dataclass
class acre_consistent_dataset:
    dataset: str = "acre_consistent_dataset"
    # file: str = "examples/acre_dataset.py"
    train_split: str = "train"
    test_split: str = "val"
    task: str = "acre_consistent"
    data_split: str = "iid"
    data_size: int = -1
    max_tokens: int = 196
    data_split_root: str = f"/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning/data/{task}/{data_split}"
    data_type: str = "language"  # ["symbolic", "language"]
    num_contexts_per_example = 2
    num_queries_per_example = 1
    num_panels_per_example = 3
