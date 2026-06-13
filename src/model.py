import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


@torch.no_grad()
def build_prompt_ensemble_prototypes(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    prompt_dict: Dict[str, List[str]],
    class_names: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Build one normalized CLIP text prototype per disease class."""
    clip_model.eval()
    class_prototypes = []

    for class_name in class_names:
        if class_name not in prompt_dict:
            raise KeyError(f"Missing text prompts for class: {class_name}")

        encoded = processor(
            text=prompt_dict[class_name],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        text_outputs = clip_model.text_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        text_embeds = clip_model.text_projection(text_outputs.pooler_output)
        text_embeds = F.normalize(text_embeds, dim=-1)

        proto = text_embeds.mean(dim=0)
        proto = F.normalize(proto, dim=0)
        class_prototypes.append(proto)

    return torch.stack(class_prototypes, dim=0)


class DiseaseSemanticPatchAttentionCLIP(nn.Module):
    """
    Text-guided CLIP model from the training notebook.

    The model computes class-conditioned patch attention where each disease text
    prototype acts as a query over CLIP ViT patch embeddings. The Streamlit app
    visualizes these attention weights directly instead of using Grad-CAM/LIME.
    """

    def __init__(
        self,
        clip_model: CLIPModel,
        text_prototypes: torch.Tensor,
        attn_dim: int = 256,
        dropout: float = 0.20,
        alpha_global: float = 0.30,
        unfreeze_last_n_vision_blocks: int = 1,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.alpha_global = alpha_global
        self.projection_dim = self.clip_model.config.projection_dim
        self.attn_dim = attn_dim

        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_last_n_vision_blocks > 0:
            for layer in self.clip_model.vision_model.encoder.layers[-unfreeze_last_n_vision_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

            if hasattr(self.clip_model.vision_model, "post_layernorm"):
                for param in self.clip_model.vision_model.post_layernorm.parameters():
                    param.requires_grad = True

            for param in self.clip_model.visual_projection.parameters():
                param.requires_grad = True

        self.patch_norm = nn.LayerNorm(self.projection_dim)
        self.text_norm = nn.LayerNorm(self.projection_dim)

        self.query_proj = nn.Linear(self.projection_dim, attn_dim)
        self.key_proj = nn.Linear(self.projection_dim, attn_dim)
        self.value_proj = nn.Linear(self.projection_dim, self.projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.class_bias = nn.Parameter(torch.zeros(text_prototypes.shape[0]))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

        self.register_buffer("text_prototypes", text_prototypes.clone())

    def refresh_text_prototypes(self, new_text_prototypes: torch.Tensor) -> None:
        self.text_prototypes = new_text_prototypes.clone().to(self.text_prototypes.device)

    def forward(self, pixel_values: torch.Tensor, return_attention: bool = False):
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        token_embeddings = vision_outputs.last_hidden_state
        pooled_output = vision_outputs.pooler_output
        patch_tokens = token_embeddings[:, 1:, :]

        patch_embeddings = self.clip_model.visual_projection(patch_tokens)
        patch_embeddings = self.patch_norm(patch_embeddings)
        patch_embeddings = self.dropout(patch_embeddings)

        global_embedding = self.clip_model.visual_projection(pooled_output)
        global_embedding = F.normalize(global_embedding, dim=-1)

        text_prototypes = self.text_norm(self.text_prototypes)
        text_prototypes = F.normalize(text_prototypes, dim=-1)

        queries = self.query_proj(text_prototypes)
        keys = self.key_proj(patch_embeddings)
        values = self.value_proj(patch_embeddings)

        attention_logits = torch.einsum("ca,bna->bcn", queries, keys) / math.sqrt(self.attn_dim)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        class_visual = torch.einsum("bcn,bnd->bcd", attention_weights, values)
        class_visual = F.normalize(class_visual, dim=-1)

        local_logits = torch.sum(class_visual * text_prototypes.unsqueeze(0), dim=-1)
        global_logits = torch.matmul(global_embedding, text_prototypes.t())

        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * ((1 - self.alpha_global) * local_logits + self.alpha_global * global_logits) + self.class_bias

        if return_attention:
            return {
                "logits": logits,
                "attention_weights": attention_weights,
                "local_logits": local_logits,
                "global_logits": global_logits,
                "patch_embeddings": patch_embeddings,
                "global_embedding": global_embedding,
            }
        return logits
