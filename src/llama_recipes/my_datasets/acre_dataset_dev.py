import copy
import json
import os

import numpy as np
from PIL import Image
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


def get_objects_str(objects, is_cropped_objs):
    num_objs = len(objects) if is_cropped_objs else 1
    return f"Objects: {' '.join(['<MASK>']*num_objs)}" 


def process_dataset(dataset, dataset_config, is_inference, is_cropped_objs):
    prompts, outputs, obj_counts = [],[],[]

    for ex_id, ex in tqdm(enumerate(dataset), total=len(dataset)):
        
        if ex_id == dataset_config.data_size:
            break

        base_prompt = "You are a helpful assistant that determines whether the light will be activated by the objects. Some objects can activate the light. The other objects cannot activate the light. There are three possible light states: on, off, and unknown.\n"

        base_obj_counts = []
        for i in range(dataset_config.num_contexts_per_example):
            panel = ex[i]
            light_state = "unknown" if panel["light_state"] == "undetermined" else panel["light_state"]
            base_prompt += f"{get_objects_str(panel['objects'], is_cropped_objs)}\nLight: {light_state}\n"
            num_objs = len(panel["objects"]) if is_cropped_objs else 1
            base_obj_counts.append(num_objs)

        for i in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            panel = ex[i]
            light_state = IDX_TO_LIGHT_STATE[panel["label"]]
            prompt = base_prompt + f"{get_objects_str(panel['objects'], is_cropped_objs)}\nLight:"
            prompts.append(prompt)
            outputs.append(f"{light_state}")

            num_objs = len(panel["objects"]) if is_cropped_objs else 1
            curr_obj_counts = base_obj_counts.copy()
            curr_obj_counts.append(num_objs)
            obj_counts.append(curr_obj_counts.copy())
            
    return prompts, outputs, obj_counts


def get_context_and_image(dataset_config, split, ex_id, panel_id):
        context_path = os.path.join(dataset_config.rendered_data_root, "scenes", f"ACRE_{split}_{ex_id:06}_{panel_id:02}.json")
        with open(context_path, "rb") as f:
            context = json.load(f)

        image_path = os.path.join(dataset_config.rendered_data_root, "images", f"ACRE_{split}_{ex_id:06}_{panel_id:02}.png")
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        return context, image_np


def get_panel_images(dataset_config, num_examples, split):
    # Load panels of each example
    all_panel_images = []
    for ex_id in tqdm(range(num_examples), desc=f"{split} get_panel_images"):
        
        if ex_id == dataset_config.data_size:
            break
        
        base_panel_images = []
        for context_panel_id in range(dataset_config.num_contexts_per_example):
            context, image_np = get_context_and_image(dataset_config, split, ex_id, context_panel_id)
            base_panel_images.append(image_np.copy())

        for query_panel_id in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            context, image_np = get_context_and_image(dataset_config, split, ex_id, query_panel_id)
            curr_panel_images = base_panel_images.copy()
            curr_panel_images.append(image_np.copy())
            all_panel_images.append(curr_panel_images.copy())
        
    return all_panel_images


def get_cropped_objects(dataset_config, num_examples, split):
    # Load scenes and corresponding cropped objects
    all_cropped_objs = []
    for ex_id in tqdm(range(num_examples), desc=f"{split} get_cropped_objects"):
        
        if ex_id == dataset_config.data_size:
            break
        
        base_cropped_objs = []
        for context_panel_id in range(dataset_config.num_contexts_per_example):
            context, image_np = get_context_and_image(dataset_config, split, ex_id, context_panel_id)
            
            for obj_ctx in context["objects"]:
                bottom, left, obj_h, obj_w = np.array(obj_ctx["bbox"])
                top = bottom + obj_h
                right = left + obj_w
                cropped_obj = image_np[left:right, bottom:top]
                base_cropped_objs.append(cropped_obj.copy())

        for query_panel_id in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            context, image_np = get_context_and_image(dataset_config, split, ex_id, query_panel_id)
            
            curr_cropped_objs = base_cropped_objs.copy()
            for obj_ctx in context["objects"]:
                bottom, left, obj_h, obj_w = np.array(obj_ctx["bbox"])
                top = bottom + obj_h
                right = left + obj_w
                cropped_obj = image_np[left:right, bottom:top]
                curr_cropped_objs.append(cropped_obj)
            all_cropped_objs.append(curr_cropped_objs.copy())

    return all_cropped_objs


def process_objects(split, all_cropped_objs, image_preprocess, image_model=None, image_model_name=None, device=None):
    res = []
    for cropped_objs in tqdm(all_cropped_objs, desc=f"{split} process_objects"):
        curr_objs = []
        for cropped_obj in cropped_objs:
            if image_model:
                # If image_model is passed in and we do not train image encoder, it means we can precompute image features
                if image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                    curr_obj = image_preprocess(Image.fromarray(cropped_obj)).unsqueeze(0)
                    with torch.no_grad():
                        image_feature = image_model.encode_image(curr_obj.to(device))
                elif image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
                    curr_obj = image_preprocess(images=Image.fromarray(cropped_obj), return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        image_feature = image_model(pixel_values=curr_obj)
                        image_feature = image_feature.last_hidden_state[:,1:]  # Exclude CLS token
                        image_features = image_features.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                curr_objs.append(image_feature.detach().cpu().numpy()[0].copy())
            else:
                curr_objs.append(curr_obj.numpy()[0].copy())
        res.append(np.array(curr_objs.copy()))
    return res


def process_images(split, all_panel_imgs, image_preprocess, image_model=None, image_model_name=None, device=None):
    res = []
    for panel_imgs in tqdm(all_panel_imgs, desc=f"{split} process_images"):
        curr_imgs = []
        for panel_img in panel_imgs:
            if image_model:
                # If image_model is passed in and we do not train image encoder, it means we can precompute image features
                if image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                    curr_img = image_preprocess(Image.fromarray(panel_img)).unsqueeze(0)
                    with torch.no_grad():
                        image_feature = image_model.encode_image(curr_img.to(device))
                elif image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
                    curr_img = image_preprocess(images=Image.fromarray(panel_img), return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        image_feature = image_model(pixel_values=curr_img)
                        image_feature = image_feature.last_hidden_state[:,1:]  # Exclude CLS token
                        image_features = image_features.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                curr_imgs.append(image_feature.detach().cpu().numpy()[0].copy())
            else:
                curr_imgs.append(curr_img.numpy()[0].copy())
        res.append(np.array(curr_imgs.copy()))
    return res


class AcreDataset(Dataset):
    def __init__(
        self, 
        dataset_config, 
        tokenizer, 
        split="train", 
        is_inference=False, 
        image_model=None, 
        image_model_name=None,
        image_preprocess=None, 
        device=None, 
        is_cropped_objs=False
    ):
        self.split = split
        self.num_examples = NUM_EXAMPLES_DICT[self.split]
        self.dataset_config = dataset_config
        self.is_cropped_objs = is_cropped_objs
        self.image_preprocess = image_preprocess
        self.is_train_image_encoder = self.dataset_config.is_train_image_encoder
        self.image_model_name = image_model_name

        self.dataset = get_data(dataset_config.data_type, self.split, dataset_config)
        self.prompts, self.outputs, self.obj_counts = process_dataset(self.dataset, dataset_config, is_inference, self.is_cropped_objs)

        if self.is_cropped_objs:
            self.all_cropped_objs = get_cropped_objects(dataset_config, self.num_examples, self.split)
            if not self.is_train_image_encoder:
                # If we do not train image encoder, we can precompute image features
                self.processed_objs = process_objects(
                    self.split, 
                    self.all_cropped_objs, 
                    image_preprocess, 
                    image_model=image_model, 
                    image_model_name=self.image_model_name,
                    device=device,
                )
        else:
            self.all_panel_imgs = get_panel_images(dataset_config, self.num_examples, self.split)
            if not self.is_train_image_encoder:
                # If we do not train image encoder, we can precompute image features
                self.processed_imgs = process_images(
                    self.split, 
                    self.all_panel_imgs, 
                    image_preprocess, 
                    image_model=image_model, 
                    image_model_name=self.image_model_name,
                    device=device,
                )

        self.tokenizer = tokenizer
        self.max_tokens = dataset_config.max_tokens
        self.is_inference = is_inference

    def __len__(self):
        return len(self.prompts)

    def preprocess_image_input(self, idx):
        res = []
        images = self.all_cropped_objs[idx] if self.is_cropped_objs else self.all_panel_imgs[idx]
        for image in images:
            if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                curr_image = self.image_preprocess(Image.fromarray(image)).unsqueeze(0)
                res.append(curr_image.numpy()[0].copy())
            elif self.image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
                curr_image = self.image_preprocess(images=Image.fromarray(image), return_tensors="pt")["pixel_values"]
                res.append(curr_image.numpy()[0].copy())
        return np.array(res)

    def __getitem__(self, idx):
        if self.is_train_image_encoder:
            images = self.preprocess_image_input(idx)
        else:
            images = self.processed_objs[idx] if self.is_cropped_objs else self.processed_imgs[idx]
        object_counts = self.obj_counts[idx]

        if not self.is_inference:
            prompt, output = self.prompts[idx], self.outputs[idx]
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_ids_mask = input_ids.ge(0)
            input_ids[~input_ids_mask] = 0
            input_ids_mask = input_ids_mask.float()
        
        else:
            # Assume batch_size=1 during model inference
            prompt, output = self.prompts[idx], self.outputs[idx]
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_ids_mask = input_ids.ge(0)
            input_ids[~input_ids_mask] = 0
            input_ids_mask = input_ids_mask.float()

        return {
            "input_ids": input_ids,
            "label": self.tokenizer.encode(output, add_special_tokens=False),
            "attention_mask": input_ids_mask,
            "objects": images,
            "object_counts": object_counts,
        }


class AcreDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors=None):
        batch_size = len(features)
        batch = {}
        batch["seq_lengths"] = torch.LongTensor([len(f["input_ids"]) for f in features])
        outputs = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            # pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["input_ids"] = outputs["input_ids"].long()
        batch["attention_mask"] = outputs["attention_mask"].long()
        batch["labels"] = torch.LongTensor([f["label"][0] for f in features if len(f["label"]) == 1])

        # Prepare padded objects
        max_num_objs = max([len(f["objects"]) for f in features])
        objects_shape = (batch_size, max_num_objs) + tuple(features[0]["objects"].shape[1:])
        objects = np.zeros(objects_shape)
        for i, f in enumerate(features):
            objects[i, :len(f["objects"])] = np.array(f["objects"]).copy()
        batch["objects"] = torch.Tensor(objects)
        batch["object_counts"] = torch.LongTensor([sum(f["object_counts"]) for f in features])
        return batch 
