import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import set_seed



NUM_EXAMPLES_DICT = {
    "train": 6000,
    "val": 2000,
    "test": 2000,
}

LABEL_DICT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
}

NUM_ATTRIBUTE_CLASSES_DICT = {
    "Type": 6,
    "Color": 10,
    "Size": 6,
}


def load_images(dataset_config, split):
    all_images, all_labels = [],[]

    data_dir = os.path.join(dataset_config.data_root, dataset_config.figure_configuration)

    for i in tqdm(range(10000), desc=f"{dataset_config.figure_configuration} [{split}]"):
        
        if len(all_labels) == dataset_config.data_size:
            break

        if (split == "train") and (i % 10 < 6):
            # training set
            file_name = f"RAVEN_{i}_train"
        elif (split == "val") and (i % 10 in [6, 7]):
            # validation set
            file_name = f"RAVEN_{i}_val"
        elif (split == "test") and (i % 10 in [8, 9]):
            # test set
            file_name = f"RAVEN_{i}_test"
        else:
            continue
        curr_npz = np.load(os.path.join(data_dir, f"{file_name}.npz"))
        all_images.append(curr_npz["image"])
        all_labels.append(curr_npz["target"].item())

        # if is_load_tree:
        #     with open(os.path.join(data_dir, f"{file_name}.xml"), "r") as f:
        #         curr_xmlstring = f.read()
        #         # temp = ET.fromstring(f.readlines())

    return np.array(all_images), np.array(all_labels)


def process_dataset(dataset, dataset_config):
    all_labels = dataset.all_labels
    prompts, outputs, obj_counts = [],[],[]

    for ex_id in tqdm(range(len(dataset)), total=len(dataset)):
        
        if ex_id == dataset_config.data_size:
            break

        prompt = "You are a helpful assistant that determines the last pattern based on the given context. You need to pick the last pattern from the given candidates. The answer is A, B, C, D, E, F, G, or H.\n"

        prompt += """Context:
        <MASK>
        <MASK>
        <MASK>
        <MASK>
        <MASK>
        <MASK>
        <MASK>
        <MASK>
        Candidate:
        A) <MASK>
        B) <MASK>
        C) <MASK>
        D) <MASK>
        E) <MASK>
        F) <MASK>
        G) <MASK>
        H) <MASK>
        Answer:
        """

        prompts.append(prompt)
        outputs.append(LABEL_DICT[all_labels[ex_id]])
        obj_counts.append(16 * (dataset_config.kshot + 1))

    return prompts, outputs, obj_counts


class RavenFairDataset(Dataset):
    def __init__(
        self, 
        dataset_config, 
        tokenizer, 
        split="train", 
        image_model=None,
        image_model_name=None,
        image_preprocess=None,
        device=None,
    ):
        self.config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.image_model = image_model
        self.image_model_name = image_model_name
        self.image_preprocess = image_preprocess
        self.device = device

        self.all_images, self.all_labels = load_images(dataset_config, split)
        self.prompts, self.outputs, self.obj_counts = process_dataset(self, dataset_config)

        if not self.config.is_train_image_encoder:
            # If we do not train image encoder, we can precompute image features
            self.all_processed_images = self.precompute_image_features()

    def __len__(self):
        return len(self.all_labels)

    def precompute_image_features(self):
        assert self.image_model, "No image_model found."

        res = []
        for ex_images in tqdm(self.all_images, desc=f"{self.split} precompute_image_features"):
            curr_images = []
            for image in ex_images:
                image = Image.fromarray(image).convert("RGB")
                if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                    curr_img = self.image_preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        image_features = self.image_model.encode_image(curr_img.to(self.device))
                elif self.image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k", "cnn", "vit"]:
                    curr_img = self.image_preprocess(images=image, return_tensors="pt")["pixel_values"].to(self.device)
                    with torch.no_grad():
                        image_features = self.image_model(pixel_values=curr_img)
                        image_features = image_features.last_hidden_state[:,1:]  # Exclude CLS token
                        image_features = image_features.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                curr_images.append(image_features.detach().cpu().numpy()[0].copy())
            res.append(np.array(curr_images.copy()))
        return res

    def preprocess_image_input(self, idx):
        res = []
        images = self.all_images[idx]
        for image in images:
            image = Image.fromarray(image).convert("RGB")
            if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                curr_image = self.image_preprocess(image).unsqueeze(0)
                res.append(curr_image.numpy()[0].copy())
            elif self.image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k", "cnn", "vit"]:
                curr_image = self.image_preprocess(images=image, return_tensors="pt")["pixel_values"]
                res.append(curr_image.numpy()[0].copy())
        return np.array(res)

    def __getitem__(self, idx):
        if self.config.is_train_image_encoder:
            images = self.preprocess_image_input(idx)
        else:
            images = self.all_processed_images[idx]
        object_counts = self.obj_counts[idx]

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


class RavenFairSymbolicDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        tokenizer,
        split="train",
    ):
        self.config = dataset_config
        self.seed = self.config.seed
        self.tokenizer = tokenizer
        self.split = split

        self.data = self.load_data()
        self.all_labels = self._get_labels()
        self.all_multihot_representations_list = self.preprocess()
        self.prompts, self.outputs, self.obj_counts = process_dataset(self, dataset_config)

    def load_data(self):
        with open(os.path.join(self.config.data_root, f"{self.config.figure_configuration}.json"), "r") as f:
            all_data = json.load(f)
        
        data = []
        for i in range(len(all_data)):
            if (self.split == "train") and (i % 10 < 6):
                data.append(all_data[str(i)])
            elif (self.split == "val") and (i % 10 in [6, 7]):
                data.append(all_data[str(i)])
            elif (self.split == "test") and (i % 10 in [8, 9]):
                data.append(all_data[str(i)])      
        return data

    def _get_labels(self):
        if self.split == "train":
            seed = self.seed
        elif self.split == "val":
            seed = self.seed + 1
        elif self.split == "test":
            seed = self.seed + 2
        set_seed(seed)

        size = len(self.data) if self.config.data_size == -1 else self.config.data_size
        labels = np.random.randint(0, 8, size=size)
        return labels

    def __len__(self):
        return len(self.all_labels)

    def _switch(self, input_list, i, j):
        input_list[i], input_list[j] = input_list[j], input_list[i]
        return input_list

    def preprocess(self):
        all_multihot_representations_list = []
        for i in range(self.__len__()):
            panels = self.data[i]["rpm"]
            label = self.all_labels[i]

            curr_example = []
            for panel in panels:
                multihot_representation = torch.concatenate([
                    F.one_hot(torch.tensor(int(panel[0][attribute])), num_classes=NUM_ATTRIBUTE_CLASSES_DICT[attribute])
                    for attribute in ["Type", "Size", "Color"]
                ])
                curr_example.append(multihot_representation)
            curr_example = self._switch(curr_example, 8, 8+label)

            all_multihot_representations_list.append(torch.stack(curr_example))

        return torch.stack(all_multihot_representations_list)
        
    def __getitem__(self, idx):
        images = self.all_multihot_representations_list[idx]
        object_counts = self.obj_counts[idx]

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


class RavenFairDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features, return_tensors=None):
        batch = {}
        batch["seq_lengths"] = torch.LongTensor(np.array([len(f["input_ids"]) for f in features]))
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
        batch["labels"] = torch.LongTensor(np.array([f["label"][0] for f in features if len(f["label"]) == 1]))

        # Prepare padded objects
        batch["objects"] = torch.Tensor(np.array([f["objects"] for f in features]))
        batch["object_counts"] = torch.LongTensor(np.array([f["object_counts"] for f in features]))
        return batch 
