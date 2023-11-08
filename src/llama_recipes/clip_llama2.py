import fire
import json
import os
import sys

import clip
import numpy as np
import pickle as pkl
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from llama_recipes.my_datasets.acre_dataset import AcreDataset, AcreDataCollator
from llama_recipes.models.llama2 import ObjectCentricLlama


device = "cuda" if torch.cuda.is_available() else "cpu"


class DatasetConfigurations:
    def __init__(self):
        self.seed = 42
        self.work_root = "/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning"
        self.task = "acre_consistent"
        self.data_split = "iid"
        self.max_tokens = 312
        self.data_type = "language"
        self.kshot = 0
        self.num_contexts_per_example = 2
        self.num_queries_per_example = 1
        self.num_panels_per_example = 3
        self.data_size = -1
        self.is_cropped_objs = False


class TrainConfigurations:
    def __init__(self):
        self.model_name = ""
        self.quantization = False
        self.lr = 1e-4
        self.num_epochs = 10
        self.gradient_accumulation_steps = 16
        self.batch_size = 2


def update_dataset_config(config, task, data_type, data_size, is_cropped_objs):
    config.task = task
    config.data_type = data_type
    config.data_size = data_size
    config.is_cropped_objs = is_cropped_objs
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


def update_train_config(train_config, model_name, quantization, lr, num_epochs, gradient_accumulation_steps, batch_size, validation_steps):
    train_config.model_name = model_name
    train_config.quantization = quantization
    train_config.lr = lr
    train_config.num_epochs = num_epochs
    train_config.gradient_accumulation_steps = gradient_accumulation_steps
    train_config.batch_size = batch_size
    train_config.validation_steps = validation_steps
    return train_config


def eval(model, data, train_config, tokenizer):
    model.projection.eval()
    loader = DataLoader(data, batch_size=train_config.batch_size, collate_fn=AcreDataCollator(tokenizer), pin_memory=True, shuffle=False, drop_last=False)

    all_predictions, all_labels = [],[]
    for batch in loader:
        seq_lengths = batch["seq_lengths"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        objects = batch["objects"].to(device)
        object_counts = batch["object_counts"].to(device)

        with torch.no_grad():
            pooled_logits, _, _ = model(objects, object_counts, input_ids, labels, attention_mask, seq_lengths)
        
        preds = pooled_logits.argmax(axis=-1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        all_predictions.append(preds)
        all_labels.append(labels)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_predictions == all_labels).sum() / len(all_labels)
    return accuracy


def main(
    model_name,
    quantization,
    lr,
    num_epochs,
    gradient_accumulation_steps,
    batch_size,
    validation_steps,
    task: str=None, # Task name of ACRE dataset: ["acre", "acre_consistent"]
    data_type: str=None, # Data type of ACRE dataset: ["symbolic", "language"]
    data_size: int=-1, # Number of data for inference. -1 means all the data.
    is_cropped_objs: bool=False, # Whether to use cropped objects as image inputs
    output_dir: str=None,
    **kwargs,
):
    # Prepare configurations
    print("Prepare configurations", flush=True)
    dataset_config = DatasetConfigurations()
    dataset_config = update_dataset_config(dataset_config, task, data_type, data_size, is_cropped_objs)
    dataset_config.data_root = os.path.join(dataset_config.work_root, "data", dataset_config.task)
    dataset_config.data_split_root = os.path.join(dataset_config.data_root, dataset_config.data_split)
    dataset_config.rendered_data_root = f"/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning/ACRE/rendered_data/{dataset_config.task}/IID"

    train_config = TrainConfigurations()
    train_config = update_train_config(train_config, model_name, quantization, lr, num_epochs, gradient_accumulation_steps, batch_size, validation_steps)

    print(dataset_config.__dict__)
    print(train_config.__dict__)
    
    # Prepare tokenizer
    print("Prepare LlamaTokenizer", flush=True)
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "mask_token": "<MASK>",
    })

    # Load CLIP
    print("Load CLIP", flush=True)
    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)
    clip_model.eval()

    # Load Llama model
    print("Load LlamaForCausalLM", flush=True)
    llama_model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        load_in_8bit=True if quantization else None,
        device_map="auto" if quantization else None,
    )
    llama_model.eval()
    llama_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)
    print(f"llama_model: {llama_model.device}", flush=True)

    # Freeze llama_model
    print("Freeze llama model", flush=True)
    for name, params in llama_model.named_parameters():
        params.requires_grad = False

    # Build model and optimizer
    print("Build model and optimizer", flush=True)
    image_hidden_size = clip_model.token_embedding.embedding_dim
    model = ObjectCentricLlama(llama_model, image_hidden_size, mask_token_id=tokenizer.mask_token_id, use_visual_encoder=True)
    model.projection.to(device)

    optimizer = optim.AdamW(
        model.projection.parameters(),
        lr=train_config.lr,
        weight_decay=0.0,
    )

    # Prepare datasets
    dataset_train = AcreDataset(
        dataset_config, 
        tokenizer, 
        "train", 
        is_inference=False, 
        clip_model=clip_model,image_preprocess=preprocess, 
        device=device
    )
    dataset_valid = AcreDataset(
        dataset_config, 
        tokenizer, 
        "val", 
        is_inference=False, 
        clip_model=clip_model,image_preprocess=preprocess, 
        device=device
    )
    dataset_test = AcreDataset(
        dataset_config, 
        tokenizer, 
        "test", 
        is_inference=False, 
        clip_model=clip_model,
        image_preprocess=preprocess, 
        device=device
    )

    # Start training
    best_accuracy = float("-inf")
    best_epoch_id = 0
    loss = -1
    for epoch_id in range(train_config.num_epochs):
        print("="*70, flush=True)
        dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, collate_fn=AcreDataCollator(tokenizer), pin_memory=True, shuffle=True, drop_last=True)

        pbar = tqdm(desc=f"Training Epoch: {epoch_id + 1}", total=len(dataloader_train)//train_config.gradient_accumulation_steps)

        for step_id, batch in enumerate(dataloader_train):
            model.projection.train()

            seq_lengths = batch["seq_lengths"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            objects = batch["objects"].to(device)
            object_counts = batch["object_counts"].to(device)

            _, _, loss = model(objects, object_counts, input_ids, labels, attention_mask, seq_lengths)
            loss = loss / train_config.gradient_accumulation_steps
            loss.backward()

            if (step_id + 1) % train_config.gradient_accumulation_steps == 0 or step_id == (len(dataloader_train) - 1):
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
            
            pbar.set_description(f"| epoch_id {best_epoch_id + 1} | step {step_id}/{len(dataloader_train)} | loss {loss.detach().float()} |")

            # pooled_logits, logits, loss = model(objects, object_counts, input_ids, labels, attention_mask, seq_lengths)

        pbar.close()

        if (epoch_id + 1) % train_config.validation_steps == 0:
            accuracy = eval(model, dataset_valid, train_config, tokenizer)
            print(f"| epoch {epoch_id + 1} | valid_acc {accuracy:8.6f} |", flush=True)
            if accuracy > best_accuracy:
                print("*"*70, flush=True)
                best_accuracy = accuracy
                best_epoch_id = epoch_id
                print(f"Best checkpoint found at Epoch {epoch_id + 1} with accuracy of {accuracy:8.6f}")
                print("*"*70, flush=True)

    # accuracy_test = eval(model, dataset_test, train_config, tokenizer)        
    # print(f"Test accuracy: {accuracy_test:8.6f}", flush=True)
    print("="*70, flush=True)
    print(f"Best validation accuracy: {best_accuracy:8.6f} at epoch {best_epoch_id}")
    print("Done!", flush=True)


if __name__ == "__main__":
    fire.Fire(main)

