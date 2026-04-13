"""LoRA layer injection for DDPM UNet2DModel (CPU prototype).

This module provides the minimal LoRA infrastructure needed to adapt
a diffusers UNet2DModel (unconditional DDPM) with low-rank matrices,
as a prerequisite for SMP-LoRA defense evaluation.

LoRA injection points:
  - mid_block.attentions.0.{to_q, to_k, to_v, to_out.0}
  - down_blocks.4.attentions.0.{to_q, to_k, to_v, to_out.0}
  - up_blocks.1.attentions.0.{to_q, to_k, to_v, to_out.0}

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA wrapper for linear layers."""

    def __init__(
        self,
        original: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.empty(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return orig_out + lora_out

    def to(self, device: torch.device | str, *args, **kwargs) -> LoRALinear:
        """Move LoRA parameters to the same device as the original layer."""
        super().to(device, *args, **kwargs)
        self.original.to(device, *args, **kwargs)
        self.lora_A = self.lora_A.to(device, *args, **kwargs)
        self.lora_B = self.lora_B.to(device, *args, **kwargs)
        self.lora_dropout = self.lora_dropout.to(device, *args, **kwargs)
        return self


def inject_lora_into_unet(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> dict[str, LoRALinear]:
    """Inject LoRA layers into UNet2DModel attention blocks.

    Args:
        model: UNet2DModel from diffusers
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: LoRA dropout probability

    Returns:
        Dict of injected LoRA layers by name
    """
    injected: dict[str, LoRALinear] = {}

    injection_points = [
        "down_blocks.4.attentions.0",
        "up_blocks.1.attentions.0",
        "mid_block.attentions.0",
    ]

    for point in injection_points:
        module = model
        for name in point.split("."):
            module = getattr(module, name)

        # Handle different types of layers
        layers_to_inject = [
            ("to_q", None),
            ("to_k", None),
            ("to_v", None),
            ("to_out", 0),  # to_out is a list, we want to inject into index 0
        ]

        for layer_name, index in layers_to_inject:
            full_name_parts = point.split(".") + [layer_name]
            if index is not None:
                full_name_parts.append(str(index))
            full_name = ".".join(full_name_parts)

            # Get the original module
            if index is None:
                original_module = getattr(module, layer_name)
            else:
                original_module = getattr(module, layer_name)[index]

            # Create LoRA wrapper
            lora_module = LoRALinear(original_module, rank=rank, alpha=alpha, dropout=dropout)

            # Replace the original module
            if index is None:
                setattr(module, layer_name, lora_module)
            else:
                getattr(module, layer_name)[index] = lora_module

            injected[full_name] = lora_module

    return injected


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect all LoRA parameters from the model."""
    lora_params: list[nn.Parameter] = []

    def collect_lora_params(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, LoRALinear):
                lora_params.extend([child.lora_A, child.lora_B])
            elif isinstance(child, nn.ModuleList):
                for i, item in enumerate(child):
                    if isinstance(item, LoRALinear):
                        lora_params.extend([item.lora_A, item.lora_B])
                    else:
                        collect_lora_params(item)
            else:
                collect_lora_params(child)

    collect_lora_params(model)
    return lora_params


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Get state dict containing only LoRA parameters."""
    state_dict: dict[str, torch.Tensor] = {}

    def collect_lora_state(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                state_dict[f"{new_prefix}.lora_A"] = child.lora_A
                state_dict[f"{new_prefix}.lora_B"] = child.lora_B
            elif isinstance(child, nn.ModuleList):
                for i, item in enumerate(child):
                    item_prefix = f"{new_prefix}.{i}"
                    if isinstance(item, LoRALinear):
                        state_dict[f"{item_prefix}.lora_A"] = item.lora_A
                        state_dict[f"{item_prefix}.lora_B"] = item.lora_B
                    else:
                        collect_lora_state(item, item_prefix)
            else:
                collect_lora_state(child, new_prefix)

    collect_lora_state(model)
    return state_dict


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load LoRA parameters from state dict."""

    def load_lora_state(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                if f"{new_prefix}.lora_A" in state_dict:
                    child.lora_A.data.copy_(state_dict[f"{new_prefix}.lora_A"])
                if f"{new_prefix}.lora_B" in state_dict:
                    child.lora_B.data.copy_(state_dict[f"{new_prefix}.lora_B"])
            elif isinstance(child, nn.ModuleList):
                for i, item in enumerate(child):
                    item_prefix = f"{new_prefix}.{i}"
                    if isinstance(item, LoRALinear):
                        if f"{item_prefix}.lora_A" in state_dict:
                            item.lora_A.data.copy_(state_dict[f"{item_prefix}.lora_A"])
                        if f"{item_prefix}.lora_B" in state_dict:
                            item.lora_B.data.copy_(state_dict[f"{item_prefix}.lora_B"])
                    else:
                        load_lora_state(item, item_prefix)
            else:
                load_lora_state(child, new_prefix)

    load_lora_state(model)


def lora_injection_summary(model: nn.Module) -> dict[str, Any]:
    """Generate summary of LoRA injection."""
    layers = []
    total_original = 0
    total_lora = 0

    def collect_summary(module: nn.Module, prefix: str = "") -> None:
        nonlocal total_original, total_lora

        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                original_params = child.original.in_features * child.original.out_features
                lora_params = (child.original.in_features * child.rank) + (child.rank * child.original.out_features)
                total_original += original_params
                total_lora += lora_params
                layers.append({
                    "name": new_prefix,
                    "in_features": child.original.in_features,
                    "out_features": child.original.out_features,
                    "rank": child.rank,
                    "alpha": child.alpha,
                    "original_params": original_params,
                    "lora_params": lora_params,
                    "compression_ratio": original_params / lora_params if lora_params > 0 else float("inf"),
                })
            elif isinstance(child, nn.ModuleList):
                for i, item in enumerate(child):
                    item_prefix = f"{new_prefix}.{i}"
                    if isinstance(item, LoRALinear):
                        original_params = item.original.in_features * item.original.out_features
                        lora_params = (item.original.in_features * item.rank) + (item.rank * item.original.out_features)
                        total_original += original_params
                        total_lora += lora_params
                        layers.append({
                            "name": item_prefix,
                            "in_features": item.original.in_features,
                            "out_features": item.original.out_features,
                            "rank": item.rank,
                            "alpha": item.alpha,
                            "original_params": original_params,
                            "lora_params": lora_params,
                            "compression_ratio": original_params / lora_params if lora_params > 0 else float("inf"),
                        })
                    else:
                        collect_summary(item, item_prefix)
            else:
                collect_summary(child, new_prefix)

    collect_summary(model)
    return {
        "num_lora_layers": len(layers),
        "total_original_params": total_original,
        "total_lora_params": total_lora,
        "overall_compression_ratio": total_original / total_lora if total_lora > 0 else float("inf"),
        "layers": layers,
    }
