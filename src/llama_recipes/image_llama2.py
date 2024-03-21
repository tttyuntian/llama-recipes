import fire
import json
import random
import os
import sys

import clip
import numpy as np
import pickle as pkl
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, BeitImageProcessor, BeitModel, ViTImageProcessor, AutoImageProcessor

from llama_recipes.my_datasets.acre_dataset_dev import AcreDataset, AcreDataCollator
from llama_recipes.my_datasets.raven_fair_dataset import RavenFairDataset, RavenFairSymbolicDataset, RavenFairDataCollator
from llama_recipes.models.object_centric_llama import ObjectCentricLlama, CnnModel, VitModel
from llama_recipes.utils import set_seed


device = "cuda" if torch.cuda.is_available() else "cpu"


class DatasetConfigurations:
    def __init__(self, task):
        if task in ["acre", "acre_consistent"]:
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
            self.is_train_image_encoder = None
        elif task == "raven_fair":
            self.seed = 42
            self.data_root = "/users/tyun/data/tyun/llm_causal_reasoning/RAVEN_FAIR/data/RAVEN-F"
            self.figure_configuration = None # ["center_single", "in_distribute_four_out_center_single", "distribute_four", "distribute_nine", "up_center_single_down_center_single", "in_center_single_out_center_single", "left_center_single_right_center_single"]
            self.kshot = 0
            self.data_size = -1
            self.is_train_image_encoder = None
        else:
            raise(f"Not supported dataset: {task}")


class TrainConfigurations:
    def __init__(self):
        self.model_name = ""
        self.quantization = False
        self.lr = 1e-4
        self.num_epochs = 10
        self.gradient_accumulation_steps = 16
        self.batch_size = 2
        self.is_train_image_encoder = False
        self.output_dir = None
        self.num_workers = 0


def update_dataset_config(config, task, data_type, data_size, is_cropped_objs, image_model_name, is_train_image_encoder, is_slot_attention, num_slots, is_symbolic_representations):
    config.task = task
    config.data_type = data_type
    config.data_size = data_size
    config.is_cropped_objs = is_cropped_objs
    config.image_model_name = image_model_name
    config.is_train_image_encoder = is_train_image_encoder
    config.is_slot_attention = is_slot_attention
    config.num_slots = num_slots
    config.is_symbolic_representations = is_symbolic_representations
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


def update_train_config(train_config, llama_model_name, model_name, quantization, lr, num_epochs, gradient_accumulation_steps, batch_size, validation_steps, image_model, num_layers, num_attention_heads, image_model_hidden_size, image_model_intermediate_size, num_workers, is_train_image_encoder, is_slot_attention, num_slots, slot_hidden_dim, slot_iters, patch_size, is_symbolic_representations, output_dir):
    train_config.llama_model_name = llama_model_name
    train_config.model_name = model_name
    train_config.quantization = quantization
    train_config.lr = lr
    train_config.num_epochs = num_epochs
    train_config.gradient_accumulation_steps = gradient_accumulation_steps
    train_config.batch_size = batch_size
    train_config.validation_steps = validation_steps
    train_config.is_train_image_encoder = is_train_image_encoder
    train_config.num_workers = num_workers
    train_config.image_model = image_model
    train_config.num_layers = num_layers
    train_config.num_attention_heads = num_attention_heads
    train_config.image_model_hidden_size = image_model_hidden_size
    train_config.image_model_intermediate_size = image_model_intermediate_size
    train_config.is_slot_attention = is_slot_attention
    train_config.num_slots = num_slots
    train_config.slot_hidden_dim = slot_hidden_dim
    train_config.slot_iters = slot_iters
    train_config.patch_size = patch_size
    train_config.is_symbolic_representations = is_symbolic_representations
    train_config.output_dir = output_dir
    return train_config


def eval(model, data, task, train_config, tokenizer):
    if train_config.is_train_image_encoder:
        model.image_model.eval()
        model.projection.eval()
        if train_config.is_slot_attention:
            model.slot_projection.eval()
            model.slot_attn.eval()
    else:
        model.projection.eval()

    if task in ["acre", "acre_consistent"]:
            loader = DataLoader(data, batch_size=train_config.batch_size, collate_fn=AcreDataCollator(tokenizer), pin_memory=True, shuffle=False, drop_last=False, num_workers=train_config.num_workers)
    elif task == "raven_fair":
        loader = DataLoader(data, batch_size=train_config.batch_size, collate_fn=RavenFairDataCollator(tokenizer), pin_memory=True, shuffle=False, drop_last=False, num_workers=train_config.num_workers)

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
    llama_model_name,
    model_name,
    quantization,
    lr,
    num_epochs,
    gradient_accumulation_steps,
    batch_size,
    validation_steps,
    image_model,
    num_layers: int=0, # Determines number of layers for CNN/ViT image encoder
    num_attention_heads: int=4, # Number of attention heads for ViT image encoder
    image_model_hidden_size: int=768, # Hidden size for CNN/ViT image encoder
    image_model_intermediate_size: int=3072, # Intermediate size for CNN/ViT image encoder
    num_workers: int=0,
    is_train_image_encoder: bool=True, # Whether we train image encoder. If `True`, LLAMA is frozen and image encoder will be trained. If `False`, both LLAMA and image encoder will be frozen.
    is_slot_attention: bool=True, # Whether to use slot attention layer on top of image encoder's outputs
    num_slots: int=-1, # Number of slots in slot attention layer
    slot_hidden_dim: int=-1, # Size of hidden dimension of slot attention layer
    slot_iters: int=-1, # Number of iterations of slot attention layer
    task: str=None, # Task name of supported dataset: ["acre", "acre_consistent", "raven_fair"]
    data_type: str=None, # Data type of ACRE dataset: ["symbolic", "language"]
    data_size: int=-1, # Number of data for inference. -1 means all the data.
    is_cropped_objs: bool=False, # Whether to use cropped objects as image inputs
    figure_configuration: str=None, # Figure configuration for RAVEN-FAIR dataset
    image_height: int=224, # Height and width of images
    patch_size: int=16, # Patch size of image model
    is_symbolic_representations: bool=None, # Whether to use symbolic representations as image inputs
    output_dir: str=None,
    **kwargs,
):
    # Prepare configurations
    print("Prepare configurations", flush=True)

    train_config = TrainConfigurations()
    train_config = update_train_config(train_config, llama_model_name, model_name, quantization, lr, num_epochs, gradient_accumulation_steps, batch_size, validation_steps, image_model, num_layers, num_attention_heads, image_model_hidden_size, image_model_intermediate_size, num_workers, is_train_image_encoder, is_slot_attention, num_slots, slot_hidden_dim, slot_iters, patch_size, is_symbolic_representations, output_dir)

    dataset_config = DatasetConfigurations(task)
    if task in ["acre", "acre_consistent"]:
        dataset_config = update_dataset_config(dataset_config, task, data_type, data_size, is_cropped_objs, image_model, is_train_image_encoder, is_slot_attention, num_slots, is_symbolic_representations)
        dataset_config.data_root = os.path.join(dataset_config.work_root, "data", dataset_config.task)
        dataset_config.data_split_root = os.path.join(dataset_config.data_root, dataset_config.data_split)
        dataset_config.rendered_data_root = f"/users/tyun/data/tyun/llm_causal_reasoning/llm_causal_reasoning/ACRE/rendered_data/{dataset_config.task}/IID"

        train_config.output_path = os.path.join(
            train_config.output_dir, 
            f"{dataset_config.task}_{dataset_config.data_split}_{dataset_config.data_type}",
            f"{train_config.llama_model_name}|T{1 if train_config.is_train_image_encoder else 0}-S{1 if train_config.is_symbolic_representations else 0}-{train_config.image_model}-{train_config.num_layers}|ImgH-{dataset_config.image_height}|P-{train_config.patch_size}|S{1 if train_config.is_slot_attention else 0}-N{train_config.num_slots}-D{train_config.slot_hidden_dim}-I{train_config.slot_iters}|lr-{train_config.lr}|e-{train_config.num_epochs}|bs-{train_config.batch_size * train_config.gradient_accumulation_steps}|",
        )

    elif task == "raven_fair":
        dataset_config.task = task
        dataset_config.figure_configuration = figure_configuration
        dataset_config.data_size = data_size
        dataset_config.image_model = image_model
        dataset_config.is_train_image_encoder = is_train_image_encoder
        dataset_config.image_height = image_height
        dataset_config.is_symbolic_representations = is_symbolic_representations

        train_config.output_path = os.path.join(
            train_config.output_dir, 
            f"{dataset_config.task}_{dataset_config.figure_configuration}",
            f"{train_config.llama_model_name}|T{1 if train_config.is_train_image_encoder else 0}-Sym{1 if train_config.is_symbolic_representations else 0}-{train_config.image_model}-{train_config.num_layers}|ImgH-{dataset_config.image_height}|P-{train_config.patch_size}|S{1 if train_config.is_slot_attention else 0}-N{train_config.num_slots}-D{train_config.slot_hidden_dim}-I{train_config.slot_iters}|lr-{train_config.lr}|e-{train_config.num_epochs}|bs-{train_config.batch_size * train_config.gradient_accumulation_steps}|",
        )
    else:
        raise(f"Not supported dataset: {task}")

    
    os.makedirs(train_config.output_path, exist_ok=True)

    train_config.ckpt_output_path = os.path.join(train_config.output_path, "image_encoder.pt")

    set_seed(dataset_config.seed)
    print(dataset_config.__dict__)
    print(train_config.__dict__)
    
    # Prepare tokenizer
    print("Prepare LlamaTokenizer", flush=True)
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "mask_token": "<MASK>",
    })

    # Load image model
    print("Load image model", flush=True)
    if train_config.is_train_image_encoder:
        if train_config.image_model in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
            image_model, image_preprocess = clip.load(train_config.image_model, device=device, jit=False)  # need to set  `jit=False` for training
        elif train_config.image_model in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
            image_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k").to(device)
            image_preprocess = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        elif train_config.image_model in ["cnn"]:
            image_model = CnnModel(train_config.num_layers).to(device)
            # image_preprocess = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
            image_preprocess = ViTImageProcessor(size={"height": dataset_config.image_height, "width": dataset_config.image_height})
        elif train_config.image_model in ["vit"]:
            image_model = VitModel(
                num_layers=train_config.num_layers, 
                num_attention_heads=train_config.num_attention_heads, 
                hidden_size=train_config.image_model_hidden_size, 
                intermediate_size=train_config.image_model_intermediate_size,
                image_size=dataset_config.image_height,
                patch_size=train_config.patch_size,
            ).to(device)
            # image_preprocess = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            image_preprocess = ViTImageProcessor(size={"height": dataset_config.image_height, "width": dataset_config.image_height})

    else:
        if train_config.image_model in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
            image_model, image_preprocess = clip.load(train_config.image_model, device=device)
        elif train_config.image_model in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
            image_model = BeitModel.from_pretrained(train_config.image_model).to(device)
            image_preprocess = BeitImageProcessor.from_pretrained(train_config.image_model)
        elif train_config.image_model in ["cnn"]:
            image_model = CnnModel(train_config.num_layers).to(device)
            # image_preprocess = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
            image_preprocess = ViTImageProcessor(size={"height": dataset_config.image_height, "width": dataset_config.image_height})
        elif train_config.image_model in ["vit"]:
            image_model = VitModel(
                num_layers=train_config.num_layers, 
                num_attention_heads=train_config.num_attention_heads, 
                hidden_size=train_config.image_model_hidden_size, 
                intermediate_size=train_config.image_model_intermediate_size,
                image_size=dataset_config.image_height,
                patch_size=train_config.patch_size,
            ).to(device)
            # image_preprocess = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            image_preprocess = ViTImageProcessor(size={"height": dataset_config.image_height, "width": dataset_config.image_height})
        
        if train_config.is_symbolic_representations:
            image_model = None
            image_preprocess = None

    if image_model:
        image_model.eval()

    # If we train image_model, freeze image_model all layers except the last
    if train_config.is_train_image_encoder:
        if train_config.image_model in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
            print("Freeze clip image encoder, except last layer", flush=True)
            for name, params in image_model.named_parameters():
                if not name.startswith("visual.transformer.resblocks.23"):
                    params.requires_grad = False
        elif train_config.image_model in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
            print("Freeze beit image encoder, except last layer", flush=True)
            for name, params in image_model.named_parameters():
                if not name.startswith("encoder.layer.11"):
                    params.requires_grad = False
        elif train_config.image_model in ["cnn", "vit"]:
            # Train everything for randomly initialized CNNs/ViTs
            pass

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
    if train_config.image_model in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
        image_hidden_size = image_model.token_embedding.embedding_dim
    elif train_config.image_model in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
        image_hidden_size = image_model.config.hidden_size
    elif train_config.image_model == "cnn":
        image_hidden_size = image_model.image_hidden_size
    elif train_config.image_model == "vit":
        image_hidden_size = image_model.config.hidden_size
    
    if train_config.is_symbolic_representations:
        if dataset_config.task in ["acre", "acre_consistent"]:
            image_hidden_size = 48
        elif dataset_config.task in ["raven_fair"]:
            if dataset_config.figure_configuration == "center_single":
                image_hidden_size = 22

    model = ObjectCentricLlama(
        llama_model, 
        image_model, 
        train_config.image_model, 
        image_hidden_size, 
        mask_token_id=tokenizer.mask_token_id, 
        use_visual_encoder=True, 
        is_train_image_encoder=train_config.is_train_image_encoder,
        is_symbolic_representations=train_config.is_symbolic_representations,
        is_slot_attention=train_config.is_slot_attention,
        num_slots=train_config.num_slots,
        slot_hidden_dim=train_config.slot_hidden_dim,
        slot_iters=train_config.slot_hidden_dim,
    )
    
    if train_config.is_train_image_encoder:
        if train_config.image_model in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "microsoft/beit-base-patch16-224-pt22k-ft22k"]:
            params = list(model.image_model.parameters()) + list(model.projection.parameters())
            model.projection.to(device)
            optimizer = optim.AdamW(
                params,
                lr=train_config.lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.2,
            )
        elif train_config.image_model in ["cnn", "vit"]:
            if train_config.is_slot_attention:
                params = list(model.image_model.parameters()) + list(model.projection.parameters()) + list(model.slot_projection.parameters()) + list(model.slot_attn.parameters())
                model.projection.to(device)
                model.slot_projection.to(device)
                model.slot_attn.to(device)
            else:
                params = list(model.image_model.parameters()) + list(model.projection.parameters())
                model.projection.to(device)
            optimizer = optim.AdamW(
                params,
                lr=train_config.lr,
                betas=(0.9, 0.98),
                eps=1e-3,
                weight_decay=0.2,
            )
    else:
        model.projection.to(device)
        optimizer = optim.AdamW(
            model.projection.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )

    if train_config.is_train_image_encoder:
        model.image_model.train()
        model.projection.train()
        if train_config.is_slot_attention:
            model.slot_projection.train()
            model.slot_attn.train()
    else:
        model.projection.train()
        
    # Prepare datasets
    if task in ["acre", "acre_consistent"]:
        dataset_train = AcreDataset(
            dataset_config, 
            tokenizer, 
            "train", 
            is_inference=False, 
            image_model=image_model,
            image_model_name=train_config.image_model,
            image_preprocess=image_preprocess, 
            device=device,
            is_cropped_objs=dataset_config.is_cropped_objs,
            is_slot_attention=dataset_config.is_slot_attention, 
            num_slots=dataset_config.num_slots,
        )
        dataset_valid = AcreDataset(
            dataset_config, 
            tokenizer, 
            "val", 
            is_inference=False, 
            image_model=image_model,
            image_model_name=train_config.image_model,
            image_preprocess=image_preprocess, 
            device=device,
            is_cropped_objs=dataset_config.is_cropped_objs,
            is_slot_attention=dataset_config.is_slot_attention, 
            num_slots=dataset_config.num_slots,
        )
        dataset_test = AcreDataset(
            dataset_config, 
            tokenizer, 
            "test", 
            is_inference=False, 
            image_model=image_model,
            image_model_name=train_config.image_model,
            image_preprocess=image_preprocess, 
            device=device,
            is_cropped_objs=dataset_config.is_cropped_objs,
            is_slot_attention=dataset_config.is_slot_attention, 
            num_slots=dataset_config.num_slots,
        )
    elif task == "raven_fair":
        if train_config.is_symbolic_representations:
            dataset_train = RavenFairSymbolicDataset(
                dataset_config, 
                tokenizer, 
                split="train",
            )
            dataset_valid = RavenFairSymbolicDataset(
                dataset_config, 
                tokenizer, 
                split="val",
            )
            dataset_test = RavenFairSymbolicDataset(
                dataset_config, 
                tokenizer, 
                split="test",
            )
        else:
            dataset_train = RavenFairDataset(
                dataset_config, 
                tokenizer, 
                split="train",
                image_model=image_model,
                image_model_name=train_config.image_model,
                image_preprocess=image_preprocess, 
                device=device,
            )
            dataset_valid = RavenFairDataset(
                dataset_config, 
                tokenizer, 
                split="val",
                image_model=image_model,
                image_model_name=train_config.image_model,
                image_preprocess=image_preprocess, 
                device=device,
            )
            dataset_test = RavenFairDataset(
                dataset_config, 
                tokenizer, 
                split="test",
                image_model=image_model,
                image_model_name=train_config.image_model,
                image_preprocess=image_preprocess, 
                device=device,
            )
    print(f"dataset_train: {len(dataset_train)}")
    print(f"dataset_valid: {len(dataset_valid)}")
    print(f"dataset_test: {len(dataset_test)}")

    # Start training
    best_accuracy = float("-inf")
    best_epoch_id = 0
    loss = -1
    for epoch_id in range(train_config.num_epochs):
        print("="*70, flush=True)

        if task in ["acre", "acre_consistent"]:
            dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, collate_fn=AcreDataCollator(tokenizer), pin_memory=True, shuffle=True, drop_last=True, num_workers=train_config.num_workers)
        elif task == "raven_fair":
            dataloader_train = DataLoader(dataset_train, batch_size=train_config.batch_size, collate_fn=RavenFairDataCollator(tokenizer), pin_memory=True, shuffle=True, drop_last=True, num_workers=train_config.num_workers)

        pbar = tqdm(desc=f"Training Epoch: {epoch_id + 1}", total=len(dataloader_train)//train_config.gradient_accumulation_steps)

        for step_id, batch in enumerate(dataloader_train):
            if train_config.is_train_image_encoder:
                model.image_model.train()
                model.projection.train()
                if train_config.is_slot_attention:
                    model.slot_projection.train()
                    model.slot_attn.train()
            else:
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
            
            pbar.set_description(f"| epoch_id {best_epoch_id + 1} | step {step_id + 1}/{len(dataloader_train)} | loss {loss.detach().float()} |")

            # pooled_logits, logits, loss = model(objects, object_counts, input_ids, labels, attention_mask, seq_lengths)

        pbar.close()

        if (epoch_id + 1) % train_config.validation_steps == 0:
            accuracy = eval(model, dataset_valid, task, train_config, tokenizer)
            print(f"| epoch {epoch_id + 1} | valid_acc {accuracy:8.6f} |", flush=True)
            if accuracy > best_accuracy:
                print("*"*70, flush=True)
                best_accuracy = accuracy
                best_epoch_id = epoch_id
                print(f"Best checkpoint found at Epoch {epoch_id + 1} with accuracy of {accuracy:8.6f}")
                print("*"*70, flush=True)

                # Save image encoder
                image_encoder_state_dict = {}
                for k, params in model.state_dict().items():
                    if not k.startswith("llama_model"):
                        image_encoder_state_dict[k] = params
                torch.save(image_encoder_state_dict, train_config.ckpt_output_path)

    # Test with best checkpoint
    best_image_encoder_state_dict = torch.load(train_config.ckpt_output_path)
    model.load_state_dict(best_image_encoder_state_dict, strict=False)
    accuracy_test = eval(model, dataset_test, task, train_config, tokenizer)        
    print("="*70, flush=True)
    print(f"Best validation accuracy: {best_accuracy:8.6f} at epoch {best_epoch_id}")
    print(f"Test accuracy: {accuracy_test:8.6f}", flush=True)
    print(f"Checkpoint saved to {train_config.output_path}")
    print("Done!", flush=True)


if __name__ == "__main__":
    fire.Fire(main)

