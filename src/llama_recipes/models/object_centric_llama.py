import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import transformers
from transformers import ViTConfig, ViTModel

from llama_recipes.models.slot_attention import SlotAttention


class ObjectCentricLlama(nn.Module):
    def __init__(
        self, 
        llama_model, 
        image_model, 
        image_model_name, 
        image_hidden_size, 
        mask_token_id,
        use_visual_encoder=True, 
        is_train_image_encoder=True, 
        is_symbolic_representations=False,
        is_slot_attention=True,
        num_slots=-1,
        slot_hidden_dim=-1,
        slot_iters=-1,
    ):
        super().__init__()
        self.llama_model = llama_model
        self.llama_hidden_size = llama_model.config.hidden_size

        self.image_model = image_model
        self.image_hidden_size = image_hidden_size
        self.image_model_name = image_model_name

        self.mask_token_id = mask_token_id
        self.use_visual_encoder = use_visual_encoder
        self.is_train_image_encoder = is_train_image_encoder
        self.is_symbolic_representations = is_symbolic_representations
        self.is_slot_attention = is_slot_attention

        if self.is_slot_attention:
            # NOTE: Assume this is a CNN for now
            self.num_slots = num_slots
            self.slot_hidden_dim = slot_hidden_dim
            self.slot_iters = slot_iters
            if self.image_model_name in ["cnn"]:
                """
                self.projection = nn.Linear(self.image_model.output_height*self.image_model.output_height, self.llama_hidden_size, bias=False)
                self.slot_attn = SlotAttention(
                    num_slots=self.num_slots,
                    dim=self.llama_hidden_size,
                    iters =self.slot_iters
                )
                """
                self.slot_projection = nn.Linear(self.image_model.output_height*self.image_model.output_height, self.slot_hidden_dim, bias=False)
                self.slot_attn = SlotAttention(
                    num_slots=self.num_slots,
                    dim=self.slot_hidden_dim,
                    iters =self.slot_iters
                )
                self.projection = nn.Linear(self.slot_hidden_dim, self.llama_hidden_size, bias=False)
            self.forward = self.forward_slot_attention
        else:
            if not self.is_train_image_encoder:
                # If we don't train image encoder, then we have an MLP projection
                # This works with precomputed features from pretrained visual encoder, or with symbolic representations of panels
                self.projection = nn.Sequential(
                    nn.Linear(self.image_hidden_size, self.llama_hidden_size, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.llama_hidden_size, self.llama_hidden_size, bias=False)
                ).float()
                self.forward = self.forward_with_precomputed_features
            else:
                # If we train image encoder, we have a linear projection or a slot attention

                # self.projection = SlotAttention(
                #     num_slots=4,
                #     dim=self.llama_hidden_size,
                #     iters=3,
                #     eps=1e-8,
                #     hidden_dim=self.llama_hidden_size,
                # )
                self.projection = nn.Linear(self.image_hidden_size, self.llama_hidden_size, bias=False)
                if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
                    self.projection = self.projection.half()
                self.forward = self.forward_original

    def forward_slot_attention(self, objects, object_counts, input_ids, labels, attention_mask, seq_lengths):
        """
        objects: [batch_size, max_num_objects, preprocessed_dimensions]
        """
        batch_size, num_objects, height = objects.shape[0], objects.shape[1], objects.shape[-1]
        
        object_embs = []
        if self.image_model_name in ["cnn"]:
            objects = objects.view(-1, 3, height, height)  # [num_images, 3, height, height]
            for obj in objects:  # [3, height, height]
                object_emb = self.image_model(obj.unsqueeze(0))  # [1, output_channels, conv_height, conv_width]
                object_emb = object_emb.flatten(start_dim=-2, end_dim=-1)  # [1, output_channels, conv_height * conv_width]
                object_emb = self.slot_projection(object_emb)  # [1, output_channels, slot_hidden_dim]
                object_emb = self.slot_attn(object_emb)  # [1, num_slots, slot_hidden_dim]
                object_embs.append(object_emb)
            
            # NOTE: num_objects == num_images == 7, which is number of panels in an example
            object_embs = torch.concatenate(object_embs)  # [num_images, num_slots, slot_hidden_dim]
            object_embs = object_embs.view(batch_size, num_objects * self.num_slots, self.slot_hidden_dim)  # [batch_size, num_objects * num_slots, slot_hidden_dim]
            object_embs = self.projection(object_embs)  # [batch_ize, num_object * num_slot, llama_hidden_size]
        
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

        # torch.cuda.empty_cache()
        outputs = self.llama_model(**batch)
        logits = outputs.logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lengths - 1]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.llama_model.lm_head.out_features), labels.view(-1))

        return pooled_logits, logits, loss


    def forward_original(self, objects, object_counts, input_ids, labels, attention_mask, seq_lengths):
        """
        objects: [batch_size, max_num_objects, preprocessed_dimensions]
        """
        batch_size, num_objects, height = objects.shape[0], objects.shape[1], objects.shape[-1]
        
        object_embs = []
        if self.image_model_name in ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]:
            objects = objects.view(-1, 3, height, height)
            for obj in objects:
                object_emb = self.image_model.encode_image(obj.unsqueeze(0))
                object_embs.append(object_emb)
            object_embs = torch.concatenate(object_embs)
            object_embs = object_embs.view(batch_size, num_objects, -1)
        elif self.image_model_name in ["microsoft/beit-base-patch16-224-pt22k-ft22k"]:
            objects = objects.view(-1, 3, height, height)
            for obj in objects:
                object_emb = self.image_model(pixel_values=obj.unsqueeze(0))
                object_emb = object_emb.last_hidden_state[:, 1:]  # Exclude CLS token
                object_emb = object_emb.mean(dim=1)  # Aggregate hidden representations to obtain image representation
                object_embs.append(object_emb)
            object_embs = torch.concatenate(object_embs)
            object_embs = object_embs.view(batch_size, num_objects, -1)
        elif self.image_model_name in ["cnn"]:
            objects = objects.view(-1, 3, height, height)
            object_embs = self.image_model(objects)  # [num_images, output_channels, conv_height, conv_width]
            object_embs = object_embs.flatten(start_dim=1, end_dim=-1)  # [num_images, output_channels * conv_height * conv_width]
            object_embs = object_embs.view(batch_size, num_objects, -1)  # [batch_size, num_objects, image_hidden_dim]
        elif self.image_model_name in ["vit"]:
            objects = objects.view(-1, 3, height, height)
            object_embs = self.image_model(objects).last_hidden_state[:,0,:]  # [num_images, image_hidden_dim]
            object_embs = object_embs.view(batch_size, num_objects, -1)  # [batch_size, num_objects, image_hidden_dim]
            
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

        # torch.cuda.empty_cache()
        outputs = self.llama_model(**batch)
        logits = outputs.logits
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), seq_lengths - 1]

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.llama_model.lm_head.out_features), labels.view(-1))

        return pooled_logits, logits, loss


    def forward_with_precomputed_features(self, objects, object_counts, input_ids, labels, attention_mask, seq_lengths):
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


class VitModel(nn.Module):
    def __init__(self, num_layers, num_attention_heads, hidden_size, intermediate_size, image_size, patch_size):
        super().__init__()
        self.config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermeidate_size=intermediate_size, 
            image_size=image_size,
            patch_size=patch_size,
        )
        self.vit = ViTModel(self.config)
        
    def forward(self, inputs, output_attentions=False):
        outputs = self.vit(pixel_values=inputs, output_attentions=output_attentions)
        return outputs
        


class CnnModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        if num_layers == 3:
            self.intermediate_output_channels = 16
            self.output_channels = 8
            self.output_height = 28  # 28 = 224 // 2^3
            self.cnn = nn.Sequential(
                nn.Conv2d(3, self.intermediate_output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.intermediate_output_channels, self.intermediate_output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.intermediate_output_channels, self.output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.image_hidden_size = self.output_channels * self.output_height * self.output_height  # 6272
        elif num_layers == 4:
            self.intermediate_output_channels = 16
            self.output_channels = 16
            self.output_height = 14  # 28 = 224 // 2^4
            self.cnn = nn.Sequential(
                nn.Conv2d(3, self.intermediate_output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.intermediate_output_channels, self.intermediate_output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.intermediate_output_channels, self.intermediate_output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(self.intermediate_output_channels, self.output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.image_hidden_size = self.output_channels * self.output_height * self.output_height  # 3136

    def forward(self, inputs):
        outputs = self.cnn(inputs)
        return outputs
