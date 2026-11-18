import math
from hashlib import sha256

import torch
from torch import nn
from torch.nn import functional as F


PROTOTYPE_MODES = ("single", "uniform", "image", "patch", "local")


class TextPrototypeAdapter(nn.Module):
    """Apply spatially routed residuals to the abnormal text match."""

    def __init__(self, dimension, mode):
        super().__init__()
        if mode not in PROTOTYPE_MODES:
            raise ValueError(f"Unknown prototype mode: {mode}")
        self.mode = mode
        count = 1 if mode == "single" else 3
        self.down = nn.ModuleList(nn.Linear(dimension, 8, bias=False) for _ in range(count))
        self.up = nn.ModuleList(nn.Linear(8, dimension, bias=False) for _ in range(count))
        for layer in self.up:
            nn.init.zeros_(layer.weight)
        if mode in ("image", "patch", "local"):
            self.router = nn.Sequential(
                nn.Linear(2 * dimension, 64), nn.GELU(), nn.Linear(64, count + 1)
            )
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)

    def forward(self, patch_layers, text_features, image_features, base_logits):
        with torch.cuda.amp.autocast(enabled=False):
            abnormal = text_features[..., 1].float()
            unit_abnormal = F.normalize(abnormal, dim=-1)
            directions = [
                F.normalize(abnormal + up(down(abnormal)), dim=-1) - unit_abnormal
                for down, up in zip(self.down, self.up)
            ]
            if self.mode != "single":
                directions.insert(0, torch.zeros_like(abnormal))
            directions = torch.stack(directions, dim=-1)

            if self.mode == "image":
                image = image_features.float().unsqueeze(1)
                weights = self._route(image, torch.zeros_like(image))

            adapted_logits = []
            for patches, logits in zip(patch_layers, base_logits):
                patches = patches.float()
                scores = 100.0 * (patches @ directions)
                if self.mode in ("local", "patch"):
                    difference = torch.zeros_like(patches)
                    if self.mode == "local":
                        batch, count, dimension = patches.shape
                        side = math.isqrt(count)
                        grid = patches.transpose(1, 2).reshape(batch, dimension, side, side)
                        neighborhood = F.avg_pool2d(grid, 3, stride=1, padding=1, count_include_pad=False)
                        difference = patches - neighborhood.flatten(2).transpose(1, 2)
                    weights = self._route(patches, difference)

                if self.mode in ("single", "uniform"):
                    correction = scores.mean(dim=-1, keepdim=True)
                else:
                    correction = (weights * scores).sum(dim=-1, keepdim=True)
                abnormal_logit = (logits[..., 1:2].float() + correction).to(logits.dtype)
                adapted_logits.append(torch.cat((logits[..., :1], abnormal_logit), dim=-1))
            return adapted_logits

    def _route(self, features, difference):
        return self.router(torch.cat((features, difference), dim=-1)).softmax(dim=-1)


def prototype_baseline_signature(args):
    digest = sha256()
    with open(args.ckt_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    fields = ("model", "fusion_mode", "image_size", "prompting_type", "prompting_branch",
              "prompting_length", "prompting_depth", "use_hsf", "k_clusters")
    return {"checkpoint_sha256": digest.hexdigest(), **{field: getattr(args, field) for field in fields}}
