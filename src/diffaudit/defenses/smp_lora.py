"""SMP-LoRA (Stable Membership-Privacy-preserving LoRA) for DDPM.

Implements the min-max optimization from:
  Luo et al., "Privacy-Preserving Low-Rank Adaptation Against Membership
  Inference Attacks for Latent Diffusion Models", AAAI 2025.

Key formulas:
  MP-LoRA:  min  L_ada + lambda * G        (unstable)
  SMP-LoRA: min  L_ada / (1 - lambda*G + delta)  (stable)

Where:
  - L_ada: adaptation loss (DDPM denoising objective on private data)
  - G: MI gain of the proxy attack model
  - lambda: privacy-utility trade-off coefficient
  - delta: smoothing constant for numerical stability

This is a CPU prototype adapted from the upstream SD v1.5 implementation
to work with unconditional DDPM (UNet2DModel) on CIFAR-10.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

from diffaudit.defenses.lora_ddpm import (
    get_lora_parameters,
    inject_lora_into_unet,
    lora_injection_summary,
)


class ProxyAttackModel(nn.Module):
    """3-layer MLP proxy attack model for SMP-LoRA.

    Takes adaptation loss features as input and outputs
    membership probability (member vs non-member).
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _compute_adaptation_loss(
    model: nn.Module,
    noisy_samples: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Compute DDPM denoising loss (adaptation loss)."""
    noise_pred = model(noisy_samples, timestep=timesteps).sample
    return F.mse_loss(noise_pred, noise)


def _compute_mi_gain(
    proxy_model: ProxyAttackModel,
    member_losses: torch.Tensor,
    nonmember_losses: torch.Tensor,
) -> torch.Tensor:
    """Compute MI gain G for the proxy attack model.

    G = accuracy of the proxy model in distinguishing members from non-members.
    Higher G means the model leaks more membership information.
    """
    member_preds = torch.sigmoid(proxy_model(member_losses.unsqueeze(-1)))
    nonmember_preds = torch.sigmoid(proxy_model(nonmember_losses.unsqueeze(-1)))

    member_correct = (member_preds > 0.5).float().mean()
    nonmember_correct = (nonmember_preds <= 0.5).float().mean()

    return (member_correct + nonmember_correct) / 2.0


def _smp_lora_objective(
    adaptation_loss: torch.Tensor,
    mi_gain: torch.Tensor,
    lambda_coeff: float,
    delta: float = 1e-4,
) -> torch.Tensor:
    """Compute SMP-LoRA objective: L_ada / (1 - lambda*G + delta)."""
    denominator = 1.0 - lambda_coeff * mi_gain + delta
    denominator = torch.clamp(denominator, min=delta)
    return adaptation_loss / denominator


def _mp_lora_objective(
    adaptation_loss: torch.Tensor,
    mi_gain: torch.Tensor,
    lambda_coeff: float,
) -> torch.Tensor:
    """Compute MP-LoRA objective: L_ada + lambda * G."""
    return adaptation_loss + lambda_coeff * mi_gain


class SMPLoRATrainer:
    """SMP-LoRA training loop for DDPM.

    This implements the min-max optimization:
    1. Inner loop: train proxy attack model to maximize MI gain
    2. Outer loop: train LoRA parameters to minimize SMP-LoRA objective

    Uses the real DDPMScheduler (linear beta schedule) for proper
    noise addition consistent with DDPM training.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lambda_coeff: float = 0.5,
        delta: float = 1e-4,
        lora_lr: float = 1e-4,
        proxy_lr: float = 1e-3,
        proxy_hidden_dim: int = 256,
        proxy_steps: int = 5,
        method: str = "smp",
        ddpm_num_train_timesteps: int = 1000,
        device: str = "cpu",
    ) -> None:
        if method not in {"smp", "mp"}:
            raise ValueError(f"method must be 'smp' or 'mp', got '{method}'")

        self.model = model
        self.lambda_coeff = lambda_coeff
        self.delta = delta
        self.method = method
        self.proxy_steps = proxy_steps
        self.device = device

        self.scheduler = DDPMScheduler(
            num_train_timesteps=ddpm_num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

        self.injected = inject_lora_into_unet(
            model, rank=rank, alpha=alpha, dropout=lora_dropout
        )

        # Move model and LoRA parameters to device
        self.model = self.model.to(device)
        for name, lora_module in self.injected.items():
            lora_module = lora_module.to(device)

        self.proxy_model = ProxyAttackModel(
            input_dim=1, hidden_dim=proxy_hidden_dim
        ).to(device)

        self.lora_params = get_lora_parameters(model)
        self.lora_optimizer = torch.optim.Adam(self.lora_params, lr=lora_lr)
        self.proxy_optimizer = torch.optim.Adam(
            self.proxy_model.parameters(), lr=proxy_lr
        )

        self.log: list[dict[str, Any]] = []

    def _add_noise_with_scheduler(
        self,
        clean_samples: torch.Tensor,
        timesteps: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean samples using DDPMScheduler.

        Returns (noisy_samples, noise) following the DDPM forward process.
        """
        noise = torch.randn_like(clean_samples)
        noisy_samples = self.scheduler.add_noise(
            clean_samples, noise, timesteps
        )
        return noisy_samples, noise

    def _train_proxy_step(
        self,
        member_losses: torch.Tensor,
        nonmember_losses: torch.Tensor,
    ) -> float:
        """Train proxy attack model for one step (maximize MI gain)."""
        self.proxy_model.train()
        self.proxy_optimizer.zero_grad()

        member_preds = self.proxy_model(member_losses.unsqueeze(-1))
        nonmember_preds = self.proxy_model(nonmember_losses.unsqueeze(-1))

        member_labels = torch.ones(member_preds.shape[0], device=self.device)
        nonmember_labels = torch.zeros(nonmember_preds.shape[0], device=self.device)

        preds = torch.cat([member_preds, nonmember_preds])
        labels = torch.cat([member_labels, nonmember_labels])

        loss = F.binary_cross_entropy_with_logits(preds, labels)
        loss.backward()
        self.proxy_optimizer.step()

        return loss.item()

    def _compute_losses(
        self,
        member_batch: torch.Tensor,
        nonmember_batch: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-sample adaptation losses for member and non-member batches.

        Uses the real DDPMScheduler for proper noise addition.
        """
        self.model.eval()
        generator = torch.Generator(device=self.device)

        with torch.no_grad():
            noisy_member, member_noise = self._add_noise_with_scheduler(
                member_batch, timesteps, generator
            )
            noisy_nonmember, nonmember_noise = self._add_noise_with_scheduler(
                nonmember_batch, timesteps, generator
            )

        member_pred = self.model(noisy_member, timestep=timesteps).sample
        nonmember_pred = self.model(noisy_nonmember, timestep=timesteps).sample

        member_losses = F.mse_loss(
            member_pred, member_noise, reduction="none"
        ).view(member_batch.shape[0], -1).mean(dim=1)
        nonmember_losses = F.mse_loss(
            nonmember_pred, nonmember_noise, reduction="none"
        ).view(nonmember_batch.shape[0], -1).mean(dim=1)

        return member_losses.detach(), nonmember_losses.detach()

    def train_step(
        self,
        member_batch: torch.Tensor,
        nonmember_batch: torch.Tensor,
        timesteps: torch.Tensor,
        step: int,
    ) -> dict[str, float]:
        """Execute one SMP-LoRA training step.

        1. Compute per-sample adaptation losses
        2. Train proxy attack model (inner loop)
        3. Compute MI gain
        4. Train LoRA parameters (outer loop)
        """
        member_losses, nonmember_losses = self._compute_losses(
            member_batch, nonmember_batch, timesteps
        )

        for _ in range(self.proxy_steps):
            proxy_loss = self._train_proxy_step(member_losses, nonmember_losses)

        mi_gain = _compute_mi_gain(
            self.proxy_model, member_losses, nonmember_losses
        )

        self.model.train()
        generator = torch.Generator(device=self.device)
        noisy_member, member_noise = self._add_noise_with_scheduler(
            member_batch, timesteps, generator
        )
        adaptation_loss = _compute_adaptation_loss(
            self.model, noisy_member, timesteps, member_noise
        )

        if self.method == "smp":
            objective = _smp_lora_objective(
                adaptation_loss, mi_gain, self.lambda_coeff, self.delta
            )
        else:
            objective = _mp_lora_objective(
                adaptation_loss, mi_gain, self.lambda_coeff
            )

        self.lora_optimizer.zero_grad()
        objective.backward()
        self.lora_optimizer.step()

        record = {
            "step": step,
            "adaptation_loss": adaptation_loss.item(),
            "mi_gain": mi_gain.item(),
            "proxy_loss": proxy_loss,
            "objective": objective.item(),
            "method": self.method,
            "lambda": self.lambda_coeff,
        }
        self.log.append(record)
        return record

    def save_checkpoint(
        self,
        path: str | Path,
        include_training_log: bool = True,
    ) -> None:
        """Save LoRA weights and training log."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        from diffaudit.defenses.lora_ddpm import get_lora_state_dict

        lora_state = get_lora_state_dict(self.model)
        torch.save(lora_state, path / "lora_weights.pt")

        proxy_state = self.proxy_model.state_dict()
        torch.save(proxy_state, path / "proxy_weights.pt")

        if include_training_log:
            log_path = path / "training_log.json"
            log_path.write_text(
                json.dumps(self.log, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        summary = lora_injection_summary(self.model)
        summary_path = path / "lora_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        meta = {
            "method": self.method,
            "lambda": self.lambda_coeff,
            "delta": self.delta,
            "num_lora_layers": summary["num_lora_layers"],
            "total_lora_params": summary["total_lora_params"],
            "ddpm_num_train_timesteps": self.scheduler.config["num_train_timesteps"],
            "ddpm_beta_schedule": self.scheduler.config["beta_schedule"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = path / "checkpoint_meta.json"
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
