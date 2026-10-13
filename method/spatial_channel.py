"""Spatial/channel residual adaptation inspired by HEPE++ Eq. (3)-(8).

Applied to existing CLIP patch projections, not a reproduction of its DINO/AFS
branch. Zero-initialized eta preserves the pretrained detector at initialization.
"""
import math
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SpatialChannelAdapter(nn.Module):
    def __init__(self, channels, mode='gated', reduction=16):
        super().__init__()
        if mode not in ('attention', 'gated') or channels < 1 or reduction < 1:
            raise ValueError('Invalid spatial/channel adapter configuration')
        self.mode = mode
        self.channels = channels
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                         nn.Conv2d(hidden, channels, 1))
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.gate_mlp = None
        if mode == 'gated':
            self.gate_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(),
                                          nn.Conv2d(hidden, channels, 1))
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.zeros_(self.gate_mlp[-1].bias)
        self.eta = nn.Parameter(torch.zeros(()))
        self.enabled = True
        self.last_statistics = {}

    def forward(self, tokens):
        if not self.enabled:
            return tokens
        if tokens.ndim != 3 or tokens.shape[-1] != self.channels:
            raise ValueError('Adapter expects [B, spatial patches, channels] without CLS')
        b, n, c = tokens.shape
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError('Adapter requires a square patch grid')
        # FP32 avoids unstable small gate/eta updates in the existing AMP forward.
        with torch.cuda.amp.autocast(enabled=False):
            x = tokens.float().transpose(1, 2).reshape(b, c, side, side)
            ac = (self.channel_mlp(x.mean((2, 3), keepdim=True)) +
                  self.channel_mlp(x.amax((2, 3), keepdim=True))).sigmoid()
            spatial_input = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
            attention = ac * self.spatial_conv(spatial_input).sigmoid()
            gate = self.gate_mlp(x).sigmoid() if self.gate_mlp is not None else None
            if gate is not None:
                attention = attention * gate
            y = x + self.eta * attention * x
            result = y.flatten(2).transpose(1, 2).to(tokens.dtype)
        with torch.no_grad():
            a = F.normalize(tokens.float(), dim=-1)
            z = F.normalize(result.float(), dim=-1)
            self.last_statistics = {
                'eta': self.eta.detach().item(),
                'normalized_feature_relative_l2': (z - a).norm().item() / max(a.norm().item(), 1e-12)}
            if gate is not None:
                self.last_statistics.update(gate_mean=gate.mean().item(),
                    gate_std=gate.std(unbiased=False).item(), gate_min=gate.min().item(), gate_max=gate.max().item())
        return result


def configure_adapter(trainer, mode, layer=24, reduction=16):
    model = trainer.clip_model
    if trainer.fusion_mode != 'add':
        raise ValueError('Feature adapters require fusion_mode=add')
    if mode == 'none':
        model.spatial_channel_adapter = None
        return None
    if layer not in model.output_layers:
        raise ValueError('Adapter layer must be an extracted visual output layer')
    index = model.output_layers.index(layer)
    channels = model.patch_token_layer.head[index].out_features
    adapter = SpatialChannelAdapter(channels, mode, reduction).to(trainer.device)
    model.spatial_channel_adapter = adapter
    model.adapter_layer_index = index
    if 'spatial_channel_adapter' not in trainer.learnable_paramter_list:
        trainer.learnable_paramter_list.append('spatial_channel_adapter')
    return adapter


def validate_adapter_checkpoint(args):
    """Prevent evaluating identical-shaped weights at the wrong output layer."""
    path = Path(args.ckt_path).with_suffix('.json')
    if not path.is_file():
        raise FileNotFoundError('Feature checkpoint requires its adjacent configuration JSON: ' + str(path))
    with path.open(encoding='utf8') as f:
        meta = json.load(f)
    for key in ('feature_adapter', 'adapter_layer', 'adapter_reduction', 'model', 'image_size',
                'prompting_type', 'prompting_branch', 'prompting_depth', 'prompting_length', 'use_hsf', 'k_clusters'):
        if meta.get(key) != getattr(args, key):
            raise ValueError('Feature checkpoint configuration mismatch: ' + key)
