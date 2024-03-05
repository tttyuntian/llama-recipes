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
        image_model_name=None,
        image_preprocess=None,
    ):
        self.config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.image_model_name = image_model_name
        self.image_preprocess = image_preprocess

        self.all_images, self.all_labels = load_images(dataset_config, split)
        self.prompts, self.outputs, self.obj_counts = process_dataset(self, dataset_config)

    def __len__(self):
        return len(self.all_labels)

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
        images = self.preprocess_image_input(idx)
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
