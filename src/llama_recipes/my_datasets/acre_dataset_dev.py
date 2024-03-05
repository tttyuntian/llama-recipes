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


def get_objects_str(objects, dataset_config, is_cropped_objs, is_slot_attention, num_slots):
    if is_slot_attention:
        num_objs = num_slots
        return f"Objects: {' '.join(['<MASK>']*num_objs)}" 
    elif dataset_config.is_language_inference:
        if dataset_config.data_type == "symbolic":
            return f"Objects: {objects}"
        elif dataset_config.data_type == "language":
            num_objs = len(objects)
            if num_objs == 1:
                return f"Objects: There is {objects[0]}."
            elif num_objs == 2:
                return f"Objects: There are {objects[0]} and {objects[1]}."
            else:
                return f"Objects: There are {', '.join(objects[:-1])} and {objects[-1]}."
        else:
            raise Exception(f"{dataset_config.data_type} not supported.")
    else:
        num_objs = len(objects) if is_cropped_objs else 1
        return f"Objects: {' '.join(['<MASK>']*num_objs)}" 


def get_num_objects(panel, is_cropped_objs, is_slot_attention, num_slots):
    if is_slot_attention:
        return num_slots
    else:
        return len(panel["objects"]) if is_cropped_objs else 1


def process_dataset(dataset, dataset_config, is_inference, is_cropped_objs, is_slot_attention, num_slots, dataset_in_context_examples=None):
    prompts, outputs, obj_counts = [],[],[]

    for ex_id, ex in tqdm(enumerate(dataset), total=len(dataset)):
        
        if ex_id == dataset_config.data_size:
            break

        base_prompt = "You are a helpful assistant that determines whether the light will be activated by the objects. Some objects can activate the light. The other objects cannot activate the light. There are three possible light states: on, off, and unknown.\n"

        # Process in-context examples
        if dataset_config.kshot > 0:
            base_prompt += "Here are some examples.\n"

            in_context_example_ids = np.random.choice(range(len(dataset_in_context_examples)), size=dataset_config.kshot)
            for i, ex_id in enumerate(in_context_example_ids):
                base_prompt += f"Example {i}\n"
                ex = dataset_in_context_examples[ex_id]
                ex_contexts = ex[:dataset_config.num_contexts_per_example]

                # Prepare in-context context panels
                for panel in ex_contexts:
                    light_state = "unknown" if panel["light_state"] == "undetermined" else panel["light_state"]
                    base_prompt += f"{get_objects_str(panel['objects'], dataset_config, is_cropped_objs, is_slot_attention, num_slots)}\nLight: {light_state}\n"

                # Prepare in-context query panel
                ex_query_id = np.random.choice(range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example), size=1).item()
                ex_query = ex[ex_query_id]

                light_state = IDX_TO_LIGHT_STATE[ex_query["label"]]
                base_prompt += f"{get_objects_str(ex_query['objects'], dataset_config, is_cropped_objs, is_slot_attention, num_slots)}\nLight: {light_state}\n\n"
            base_prompt += f"Example {dataset_config.kshot}\n"

        # Process query example
        base_obj_counts = []
        for i in range(dataset_config.num_contexts_per_example):
            panel = ex[i]
            light_state = "unknown" if panel["light_state"] == "undetermined" else panel["light_state"]
            base_prompt += f"{get_objects_str(panel['objects'], dataset_config, is_cropped_objs, is_slot_attention, num_slots)}\nLight: {light_state}\n"
            num_objs = get_num_objects(panel, is_cropped_objs, is_slot_attention, num_slots)
            base_obj_counts.append(num_objs)

        for i in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            panel = ex[i]
            light_state = IDX_TO_LIGHT_STATE[panel["label"]]
            prompt = base_prompt + f"{get_objects_str(panel['objects'], dataset_config, is_cropped_objs, is_slot_attention, num_slots)}\nLight:"
            prompts.append(prompt)
            outputs.append(f"{light_state}")

            num_objs = get_num_objects(panel, is_cropped_objs, is_slot_attention, num_slots)
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
                        image_features = image_model.encode_image(curr_obj.to(device))
                elif image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k", "cnn", "vit"]:
                    curr_obj = image_preprocess(images=Image.fromarray(cropped_obj), return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        image_features = image_model(pixel_values=curr_obj)
                        image_features = image_features.last_hidden_state[:,1:]  # Exclude CLS token
                        image_features = image_features.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                curr_objs.append(image_features.detach().cpu().numpy()[0].copy())
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
                        image_features = image_model.encode_image(curr_img.to(device))
                elif image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k", "cnn", "vit"]:
                    curr_img = image_preprocess(images=Image.fromarray(panel_img), return_tensors="pt")["pixel_values"].to(device)
                    with torch.no_grad():
                        image_features = image_model(pixel_values=curr_img)
                        image_features = image_features.last_hidden_state[:,1:]  # Exclude CLS token
                        image_features = image_features.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                curr_imgs.append(image_features.detach().cpu().numpy()[0].copy())
            else:
                curr_imgs.append(curr_img.numpy()[0].copy())
        res.append(np.array(curr_imgs.copy()))
    return res


def get_multihot_panel_representation(objects):
    res = [1e-3] * 48  # there are 48 possible objects
    for obj in objects:
        res[obj] = 1
    return res


def get_symbolic_representations(data, dataset_config, num_examples, split):
    all_symbolic_representations = []
    for ex_id in tqdm(range(num_examples), desc=f"{split} get_symbolic_representations"):

        if ex_id == dataset_config.data_size:
            break

        ex = data[ex_id]
        base_panel_reprs = []
        for context_panel_id in range(dataset_config.num_contexts_per_example):
            panel = ex[context_panel_id]["objects"]
            multihot_panel_repr = get_multihot_panel_representation(panel)
            base_panel_reprs.append(multihot_panel_repr.copy())

        for query_panel_id in range(dataset_config.num_contexts_per_example, dataset_config.num_panels_per_example):
            panel = ex[query_panel_id]["objects"]
            multihot_panel_repr = get_multihot_panel_representation(panel)
            curr_panel_reprs = base_panel_reprs.copy()
            curr_panel_reprs.append(multihot_panel_repr.copy())
            all_symbolic_representations.append(curr_panel_reprs.copy())
        
    return np.array(all_symbolic_representations)


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
        is_cropped_objs=False,
        is_slot_attention=False, 
        num_slots=-1,
        is_symbolic_representations=False,
    ):
        self.split = split
        self.num_examples = NUM_EXAMPLES_DICT[self.split]
        self.dataset_config = dataset_config
        self.is_cropped_objs = is_cropped_objs
        self.image_preprocess = image_preprocess
        self.is_train_image_encoder = self.dataset_config.is_train_image_encoder
        self.image_model_name = image_model_name
        self.is_slot_attention = is_slot_attention
        self.num_slots = num_slots
        self.is_symbolic_representations = is_symbolic_representations
        self.is_inference = is_inference
        self.kshot = dataset_config.kshot

        self.data_train = None
        if self.kshot > 0:
            self.dataset_train = get_data(dataset_config.data_type, "train", dataset_config)
        
        self.dataset = get_data(dataset_config.data_type, self.split, dataset_config)
        self.prompts, self.outputs, self.obj_counts = process_dataset(
            self.dataset, 
            dataset_config, 
            self.is_inference, 
            self.is_cropped_objs, 
            self.is_slot_attention, 
            self.num_slots, 
            dataset_in_context_examples=self.dataset_train
        )

        if not self.is_inference:
            if self.is_symbolic_representations:
                # NOTE: Use symbolic representations for each panel. Only for debugging purpose.
                assert not self.is_cropped_objs, "Symbolic representations do not work with object centric representations."
                assert not self.is_train_image_encoder, "When using symbolic representations, no image encoder needs be trained."
                self.all_symbolic_representations = get_symbolic_representations(self.dataset, dataset_config, self.num_examples, self.split)
            else:
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

    def __len__(self):
        return len(self.prompts)

    def preprocess_image_input(self, idx):
        res = []
        images = self.all_cropped_objs[idx] if self.is_cropped_objs else self.all_panel_imgs[idx]
        for image in images:
            if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                curr_image = self.image_preprocess(Image.fromarray(image)).unsqueeze(0)
                res.append(curr_image.numpy()[0].copy())
            elif self.image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k", "cnn", "vit"]:
                curr_image = self.image_preprocess(images=Image.fromarray(image), return_tensors="pt")["pixel_values"]
                res.append(curr_image.numpy()[0].copy())
        return np.array(res)

    def __getitem__(self, idx):
        images = None
        if not self.is_inference:
            if self.is_train_image_encoder:
                images = self.preprocess_image_input(idx)
            else:
                if self.is_symbolic_representations:
                    images = self.all_symbolic_representations[idx]
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
            # Assume batch_size=1 during model inference with ONLY language
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
        if features[0]["objects"] is not None:
            max_num_objs = max([len(f["objects"]) for f in features])
            objects_shape = (batch_size, max_num_objs) + tuple(features[0]["objects"].shape[1:])
            objects = np.zeros(objects_shape)
            for i, f in enumerate(features):
                objects[i, :len(f["objects"])] = np.array(f["objects"]).copy()
            batch["objects"] = torch.Tensor(objects)
            batch["object_counts"] = torch.LongTensor([sum(f["object_counts"]) for f in features])
        return batch 
