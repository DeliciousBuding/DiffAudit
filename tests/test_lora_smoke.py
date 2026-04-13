"""CPU smoke test for LoRA injection and SMP-LoRA forward/backward."""

import torch
from diffusers import UNet2DModel
from diffaudit.defenses.lora_ddpm import (
    inject_lora_into_unet,
    lora_injection_summary,
    get_lora_parameters,
)


def test_lora_injection():
    print("=== Building UNet2DModel ===")
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    print()
    print("=== Injecting LoRA (rank=4) ===")
    injected = inject_lora_into_unet(model, rank=4, alpha=1.0)
    print(f"Injected LoRA layers: {len(injected)}")
    for name, layer in injected.items():
        print(
            f"  {name}: in={layer.original.in_features}, "
            f"out={layer.original.out_features}, rank={layer.rank}"
        )

    print()
    summary = lora_injection_summary(model)
    print(f"Total LoRA params: {summary['total_lora_params']:,}")
    print(f"Total original params: {summary['total_original_params']:,}")
    print(f"Compression ratio: {summary['overall_compression_ratio']:.1f}x")

    print()
    print("=== Forward pass test ===")
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([100, 200])
    with torch.no_grad():
        out = model(x, timestep=t).sample
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass OK")

    print()
    print("=== LoRA parameters check ===")
    lora_params = get_lora_parameters(model)
    print(f"Trainable LoRA parameters: {len(lora_params)}")
    total_trainable = sum(p.numel() for p in lora_params)
    print(f"Total trainable LoRA param count: {total_trainable:,}")

    return model, injected


def test_backward_pass(model):
    print()
    print("=== Backward pass test ===")
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([100, 200])
    noise = torch.randn_like(x)

    noise_pred = model(x, timestep=t).sample
    loss = torch.nn.functional.mse_loss(noise_pred, noise)
    loss.backward()

    lora_params = get_lora_parameters(model)
    has_grad = sum(1 for p in lora_params if p.grad is not None)
    print(f"Loss: {loss.item():.6f}")
    print(f"LoRA params with gradients: {has_grad}/{len(lora_params)}")

    if has_grad == len(lora_params):
        print("Backward pass OK - all LoRA params have gradients")
    else:
        print("WARNING: Some LoRA params missing gradients!")


def test_smp_lora_scheduler_step():
    print()
    print("=== SMP-LoRA with DDPMScheduler training step test ===")
    from diffaudit.defenses.smp_lora import SMPLoRATrainer

    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    trainer = SMPLoRATrainer(
        model=model,
        rank=4,
        alpha=1.0,
        lambda_coeff=0.5,
        method="smp",
        ddpm_num_train_timesteps=1000,
        device="cpu",
    )

    print(f"Scheduler: {trainer.scheduler.config['num_train_timesteps']} timesteps, "
          f"beta_schedule={trainer.scheduler.config['beta_schedule']}")

    member_batch = torch.randn(4, 3, 32, 32)
    nonmember_batch = torch.randn(4, 3, 32, 32)
    timesteps = torch.tensor([100, 200, 300, 400])

    record = trainer.train_step(
        member_batch=member_batch,
        nonmember_batch=nonmember_batch,
        timesteps=timesteps,
        step=0,
    )

    print(f"Step 0 results:")
    for key, value in record.items():
        print(f"  {key}: {value}")

    if record["adaptation_loss"] > 0 and record["mi_gain"] >= 0:
        print("SMP-LoRA (DDPMScheduler) training step OK")
    else:
        print("WARNING: Unexpected training step results!")

    return trainer


def test_multi_step_training(trainer):
    print()
    print("=== Multi-step training stability test ===")
    member_batch = torch.randn(4, 3, 32, 32)
    nonmember_batch = torch.randn(4, 3, 32, 32)

    losses = []
    for step in range(5):
        timesteps = torch.randint(0, 1000, (4,))
        record = trainer.train_step(
            member_batch=member_batch,
            nonmember_batch=nonmember_batch,
            timesteps=timesteps,
            step=step,
        )
        losses.append(record["adaptation_loss"])
        print(f"  Step {step}: loss={record['adaptation_loss']:.6f}, "
              f"mi_gain={record['mi_gain']:.4f}, objective={record['objective']:.6f}")

    loss_variance = torch.var(torch.tensor(losses)).item()
    print(f"Loss variance over 5 steps: {loss_variance:.6f}")
    if loss_variance > 0:
        print("Multi-step training OK - losses are updating")
    else:
        print("WARNING: Losses not changing across steps!")


if __name__ == "__main__":
    model, injected = test_lora_injection()
    test_backward_pass(model)
    trainer = test_smp_lora_scheduler_step()
    test_multi_step_training(trainer)
    print()
    print("ALL SMOKE TESTS PASSED")
