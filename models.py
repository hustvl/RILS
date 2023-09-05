# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from github.com/openai/CLIP
from collections import OrderedDict

import numpy as np
import timm
import torch
from torch import nn

import losses
from models_rils import rils_vit_base_patch16_dec768d1b, rils_vit_large_patch16_dec1024d1b

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cross_attention(self, q, kv):
        return self.cross_attn(q, kv, kv, need_weights=False)[0]

    def forward(self, x, kv):
        x = x + self.attention(self.ln_1(x))
        x = x + self.cross_attention(self.ln_3(x), self.ln_3(kv))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualCrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x, kv):
        for blk in self.resblocks:
            x = blk(x, kv)
        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 clip_gap=True,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width
        self.clip_gap = clip_gap

        self.visual = vision_model

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        if self.clip_gap:
            x = x[:, 1:].mean(1)
        else:
            x = x[:, 0]

        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


class ProjectionHead(nn.Module):
    def __init__(self,
                 num_layers=2,
                 in_dim=768,
                 hidden_dim=4096,
                 out_dim=32768):
        super().__init__()
        assert num_layers > 1

        layers = []
        for _ in range(num_layers):
            if _ == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            elif _ == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
                layers.append(nn.BatchNorm1d(out_dim, affine=False))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)

        self.init_std = .02
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        else:
            pass

    def forward(self, x):
        return self.layers(x)


class SIMCLR(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = self._build_mlp(in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def encode_image(self, image):
        x = self.visual(image)

        return x

    def forward(self, aug1, aug2):
        h1 = self.visual(aug1)
        h2 = self.visual(aug2)

        aug1_embed = self.image_mlp(h1)
        aug2_embed = self.image_mlp(h2)

        return {'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}


class SLIP(CLIP):
    def __init__(self,
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.image_mlp = self._build_mlp(in_dim=self.vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, image, text, aug1, aug2):
        aug1_embed = self.image_mlp(self.visual(aug1))
        aug2_embed = self.image_mlp(self.visual(aug2))
        
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}


class RILS(CLIP):
    def __init__(self,
                 mask_ratio=0.75,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.mask_ratio = mask_ratio
    
    def encode_image(self, image):
        assert not self.training
        masked_feat, masked_pred, unmasked_feat, mask = self.visual(image, mask_ratio=0.)
        image_embed = unmasked_feat[:, 1:, ...].mean(1) @ self.image_projection
        return image_embed

    def forward(self, image, text):
        masked_feat, masked_pred, unmasked_feat, mask = self.visual(image, mask_ratio=self.mask_ratio if self.training else 0.)
        image_embed = unmasked_feat[:, 1:, ...].mean(1) @ self.image_projection
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'masked_feat': masked_feat @ self.image_projection,
                'masked_pred': masked_pred @ self.image_projection,
                'unmasked_feat': unmasked_feat @ self.image_projection,
                'mask': mask}


def get_loss(model, ssl_temp, ssl_scale):
    if model.startswith('SLIP'):
        ssl_loss = losses.SIMCLRLoss(temperature=ssl_temp)
        return losses.SLIPLoss(ssl_loss, ssl_scale)
    if model.startswith('CLIP'):
        return losses.CLIPLoss()
    if model.startswith('SIMCLR'):
        return losses.SIMCLRLoss(temperature=ssl_temp)
    if model.startswith('RILS'):
        return losses.RILSLoss()
    raise NotImplementedError


def get_metric_names(model):
    if model.startswith('SLIP'):
        return ['loss', 'clip_loss', 'ssl_loss', 'clip_acc', 'ssl_acc']
    elif model.startswith('CLIP'):
        return ['loss', 'clip_loss', 'clip_acc']
    elif model.startswith('SIMCLR'):
        return ['loss', 'ssl_loss', 'ssl_acc']
    elif model.startswith('RILS'):
        return ['loss', 'clip_loss', 'clip_acc', 'recon_loss']
    else:
        raise NotImplementedError

def RILS_VITB16(**kwargs):
    vision_model = rils_vit_base_patch16_dec768d1b()
    model = RILS(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)
    
    return model

def RILS_VITL16(**kwargs):
    vision_model = rils_vit_large_patch16_dec1024d1b()
    model = RILS(embed_dim=768, vision_width=1024, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=768, transformer_heads=12, transformer_layers=12, **kwargs)
    
    return model
