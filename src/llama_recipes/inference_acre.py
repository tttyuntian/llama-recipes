# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from collections import defaultdict
import fire
import os

import pickle as pkl
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer

from llama_recipes.my_datasets.acre_dataset_dev import AcreDataset
from llama_recipes.inference.model_utils import load_model, load_peft_model


NUM_EXAMPLES_DICT = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


class DatasetConfigurations:
    def __init__(self):
        self.seed = 42
        self.work_root = "/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning"
        self.task = "acre"
        self.data_split = "iid"
        self.max_tokens = 312
        self.data_type = "language"
        self.kshot = 0
        self.num_contexts_per_example = 6
        self.num_queries_per_example = 4
        self.num_panels_per_example = 10
        self.data_size = -1
        self.is_train_image_encoder = False
        # self.train_query_type = "skip"


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


def get_query_problem_accuracy(data, responses, dataset_config):
    # Compute query accuracy
    correct = []
    for response, label in zip(responses, data.outputs):
        pred = response.split(":")[-1]
        if "on" in pred:
            pred = "on"
        elif "off" in pred:
            pred = "off"
        elif "unknown" in pred:
            pred = "unknown"
        correct.append(pred == label)
    query_accuracy = sum(correct) / len(responses)

    # Compute problem accuracy -- Each problem has 4 queries
    problem_correct = []
    for i in range(0, len(correct), dataset_config.num_queries_per_example):
        problem_correct.append(
            all(correct[i : (i + dataset_config.num_queries_per_example)])
        )
    problem_accuracy = sum(problem_correct) / (len(responses) / dataset_config.num_queries_per_example)
    
    return query_accuracy, problem_accuracy


def get_query_accuracy_by_query_type(data, responses):
    # Collect query type for each example
    types = []
    types_counter = defaultdict(int)
    for ex in data.dataset[:len(responses)]:
        queries = ex[-4:]
        for q in queries:
            types.append(q["type"])
            types_counter[q["type"]] += 1

    # Performance by query types
    correct = defaultdict(int)
    for response, label, q_type in zip(responses, data.outputs, types):
        pred = response.split(":")[-1]
        if "on" in pred:
            pred = "on"
        elif "off" in pred:
            pred = "off"
        elif "unknown" in pred:
            pred = "unknown"
        
        if pred == label:
            correct[q_type] += 1

    accuracy_by_type = {k: 0.0 for k in types_counter.keys()}
    for k, v in correct.items():
        accuracy_by_type[k] = v / types_counter[k]
    return accuracy_by_type


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
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool=False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    task: str=None, # Task name of ACRE dataset: ["acre", "acre_consistent"]
    data_type: str=None, # Data type of ACRE dataset: ["symbolic", "language"]
    data_size: int=-1, # Number of data for inference. -1 means all the data.
    # train_query_type: str="skip", # 
    output_dir: str=None,
    **kwargs
):
    # Prepare dataset configurations
    dataset_config = DatasetConfigurations()
    dataset_config = update_dataset_config(dataset_config, task, data_type, data_size)
    print(dataset_config.__dict__)

    # Prepare prompts
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    dataset_test = AcreDataset(
        dataset_config,
        tokenizer,
        split="test",
        is_inference=True,
    )
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
    for ex_id, batch in tqdm(enumerate(dataset_test), total=len(dataset_test)):

        # batch = tokenizer(user_prompt, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        input_ids = batch["input_ids"].unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # output_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0][-2], skip_special_tokens=True)
        output_text = tokenizer.decode(outputs.logits.argmax(dim=-1)[0][-1], skip_special_tokens=True)
        responses.append(output_text)

        """
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(tokenizer.convert_ids_to_tokens(outputs[0]), flush=True)
        responses.append(output_text)
        """

    # Save responses
    print("-"*50)
    print("Collected responses done!")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "responses.pkl")
    with open(output_path, "wb") as f:
        pkl.dump(responses, f)
    print(f"responses save to {output_path}")

    # Compute query/problem accuracy and query accuracy by query types
    print("-"*50)
    query_accuracy, problem_accuracy = get_query_problem_accuracy(dataset_test, responses, dataset_config)
    print(f"query_accuracy: {query_accuracy:8.6f}")
    print(f"problem_accuracy: {problem_accuracy:8.6f}")
    
    print("-"*50)
    accuracy_by_type = get_query_accuracy_by_query_type(dataset_test, responses)
    print("Query accuracy by query type:")
    for k, v in accuracy_by_type.items():
        print(f"{k}: {v:8.6f}")

    print("-"*50)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)
