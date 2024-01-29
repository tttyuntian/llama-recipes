import fire
from collections import defaultdict
import os
import sys
import time

import pickle as pkl
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer

from llama_recipes.datasets.acre_dataset import AcreDataset
from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


class DatasetConfigurations:
    def __init__(self):
        self.seed = 42
        #self.work_root = "/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning"
        self.work_root = "/oscar/data/csun45/jbyers3/acre-raw"
        self.task = "acre_consistent"
        self.data_split = "iid"
        self.max_tokens = 312
        self.data_type = "language"
        self.kshot = 0
        self.num_contexts_per_example = 2
        self.num_queries_per_example = 1
        self.num_panels_per_example = 3
        #self.data_size = -1
        self.data_size = 2

def update_dataset_config(config, task, data_type, data_size):
    config.task = task
    config.data_type = data_type
    config.data_size = data_size
    if config.task == "acre":
        config.max_tokens = 312
        config.num_contexts_per_example = 6
        config.num_queries_per_example = 4
        config.num_panels_per_example = 10
    elif config.task == "acre_consistent":
        config.max_tokens = 196
        config.num_contexts_per_example = 2
        config.num_queries_per_example = 1
        config.num_panels_per_example = 3
    config.data_root = os.path.join(config.work_root, "data", config.task)
    config.data_split_root = os.path.join(config.data_root, config.data_split)
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
    data_size: int=2, # Number of data for inference. -1 means all the data.
    output_dir: str="/results/inference",
    **kwargs
):

    dataset_config = DatasetConfigurations()
    dataset_config = update_dataset_config(dataset_config, task, data_type, data_size)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    dataset_test = AcreDataset(dataset_config, tokenizer, split="test", is_inference=True)
 
    

    with open('./results/inference/responses.pkl', 'rb') as file:
        responses = pkl.load(file)
        #print(object)
        print("collected results")

    types = []
    for ex in dataset_test.dataset:
        queries = ex[-4:]
        for q in queries:
            types.append(q["type"])

    correct = 0
    for response, label in zip(responses, dataset_test.outputs):
        pred = response.split(":")[-1]
        if "on" in pred:
            pred = "on"
        elif "off" in pred:
            pred = "off"
        elif "unknown" in pred:
            pred = "unknown"
        
        if pred == label:
            correct += 1

    print(f"accuracy: {correct / len(responses):8.6f}")


    # Performance by query types
    correct = defaultdict(int)
    for response, label, q_type in zip(responses, dataset_test.outputs, types):
        pred = response.split(":")[-1]
        if "on" in pred:
            pred = "on"
        elif "off" in pred:
            pred = "off"
        elif "unknown" in pred:
            pred1 = "unknown"
        
        if pred == label:
            correct[q_type] += 1

    for k, v in correct.items():
        print(f"{k}: {v / len(responses):8.6f}")


    #########################################################
    """
    print()
    print("train set: ")
    dataset_config.data_type = "symbolic"
    dataset_config.train_query_type = ""
    dataset_target = AcreDataset(dataset_config, tokenizer, split="train", is_inference=False)

    counter = defaultdict(lambda: defaultdict(int))
    for ex_id, ex in tqdm(enumerate(dataset_target.dataset)):
        queries = ex[-4:]
        for q in queries:
            counter[q["type"]][q["label"]] += 1

    for k, v in counter.items():
        print(f"{k} {v / sum(counter.values())}")
    """


if __name__ == "__main__":
    fire.Fire(main)