import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import transformers


class ObjectCentricLlama(nn.Module):
    def __init__(self, llama_model, image_hidden_size, mask_token_id,use_visual_encoder=True):
        super().__init__()
        self.llama_model = llama_model
        self.llama_hidden_size = llama_model.config.hidden_size
        self.image_hidden_size = image_hidden_size
        self.mask_token_id = mask_token_id
        self.use_visual_encoder = use_visual_encoder
        
        self.projection = nn.Sequential(
            nn.Linear(self.image_hidden_size, self.llama_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.llama_hidden_size, self.llama_hidden_size, bias=False)
        ).float()

        self.forward = self.forward_with_clip_features
        # if self.use_visual_encoder:
        #     self.forward = self.forward_with_visual_encoder
        # else:
        #     self.forward = self.forward_original

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
    