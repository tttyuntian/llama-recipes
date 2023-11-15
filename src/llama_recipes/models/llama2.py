import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import transformers

from llama_recipes.models.slot_attention import SlotAttention


class ObjectCentricLlama(nn.Module):
    def __init__(self, llama_model, clip_model, image_hidden_size, mask_token_id,use_visual_encoder=True, is_train_image_encoder=True):
        super().__init__()
        self.llama_model = llama_model
        self.llama_hidden_size = llama_model.config.hidden_size

        self.clip_model = clip_model
        self.image_hidden_size = image_hidden_size

        self.mask_token_id = mask_token_id
        self.use_visual_encoder = use_visual_encoder
        self.is_train_image_encoder = is_train_image_encoder
        
        if not self.is_train_image_encoder:
            # If we don't train image encoder, then we have an MLP projection
            self.projection = nn.Sequential(
                nn.Linear(self.image_hidden_size, self.llama_hidden_size, bias=False),
                nn.ReLU(),
                nn.Linear(self.llama_hidden_size, self.llama_hidden_size, bias=False)
            ).float()
            self.forward = self.forward_with_clip_features
        else:
            # If we train image encoder, we have a linear projection

            # self.projection = SlotAttention(
            #     num_slots=4,
            #     dim=self.llama_hidden_size,
            #     iters=3,
            #     eps=1e-8,
            #     hidden_dim=self.llama_hidden_size,
            # )
            self.projection = nn.Linear(self.image_hidden_size, self.llama_hidden_size, bias=False).half()
            self.forward = self.forward_original

        # if self.use_visual_encoder:
        #     self.forward = self.forward_with_visual_encoder
        # else:
        #     self.forward = self.forward_original


    def forward_original(self, objects, object_counts, input_ids, labels, attention_mask, seq_lengths):
        """
        objects: [batch_size, max_num_objects, preprocessed_dimensions]
        """
        batch_size, num_objects, height = objects.shape[0], objects.shape[1], objects.shape[-1]
        
        objects = objects.view(-1, 3, height, height)
        object_embs = []
        for obj in objects:
            object_emb = self.clip_model.encode_image(obj.unsqueeze(0))  # WARNING: hard-coded [3, 336, 336] for CLIP checkpoint `ViT-L/14@336px``
            object_embs.append(object_emb)
        object_embs = torch.concatenate(object_embs)
        object_embs =object_embs.view(batch_size, num_objects, -1)
        object_embs = self.projection(object_embs)

        inputs_embeds = []
        for i in range(batch_size):
            ex_object_embs = object_embs[i]
            ex_input_ids = input_ids[i]
            ex_object_ids = torch.where(ex_input_ids == self.mask_token_id)[0]
            ex_object_counts = object_counts[i]
            ex_input_embeds = self.llama_model.model.embed_tokens(ex_input_ids)
            ex_input_embeds[ex_object_ids] = ex_object_embs[:ex_object_counts].half()
            inputs_embeds.append(ex_input_embeds)
        inputs_embeds = torch.stack(inputs_embeds, dim=0)
        batch = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        torch.cuda.empty_cache()
        outputs = self.llama_model(**batch)
        logits = outputs.logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lengths - 1]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.llama_model.lm_head.out_features), labels.view(-1))

        return pooled_logits, logits, loss


    def forward_with_clip_features(self, objects, object_counts, input_ids, labels, attention_mask, seq_lengths):
        batch_size = objects.shape[0]
        object_embs = self.projection(objects)
        inputs_embeds = []
        for i in range(batch_size):
            ex_object_embs = object_embs[i]
            ex_input_ids = input_ids[i]
            ex_object_ids = torch.where(ex_input_ids == self.mask_token_id)[0]
            ex_object_counts = object_counts[i]
            ex_input_embeds = self.llama_model.model.embed_tokens(ex_input_ids)
            ex_input_embeds[ex_object_ids] = ex_object_embs[:ex_object_counts].half()
            inputs_embeds.append(ex_input_embeds)
        inputs_embeds = torch.stack(inputs_embeds, dim=0)

        batch = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        outputs = self.llama_model(**batch)
        logits = outputs.logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lengths - 1]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.llama_model.lm_head.out_features), labels.view(-1))

        return pooled_logits, logits, loss

    # def forward_original(self, obs_feats, input_ids, labels, attention_mask):
    #     inputs_embeds = self.llama.model.model.embed_tokens(input_ids)
    #     batch = {
    #         "inputs_embeds": inputs_embeds,
    #         "labels": labels,
    #         "attention_mask": attention_mask,
    #     }
    #     outputs = self.llama(**batch)
    #     return outputs
    