from types import SimpleNamespace

import torch
import torch.nn as nn

from diffaudit.defenses import smp_lora as smp_module
from diffaudit.defenses.smp_lora import SMPLoRATrainer
from scripts import train_smp_lora


def test_parse_args_accepts_seed(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_smp_lora.py",
            "--random_init",
            "--member_dir",
            "member",
            "--nonmember_dir",
            "nonmember",
            "--seed",
            "123",
        ],
    )

    args = train_smp_lora.parse_args()

    assert args.seed == 123


def test_parse_args_defaults_to_no_explicit_seed(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_smp_lora.py",
            "--random_init",
            "--member_dir",
            "member",
            "--nonmember_dir",
            "nonmember",
        ],
    )

    args = train_smp_lora.parse_args()

    assert args.seed is None


def test_parse_args_defaults_to_adam_optimizer(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_smp_lora.py",
            "--random_init",
            "--member_dir",
            "member",
            "--nonmember_dir",
            "nonmember",
        ],
    )

    args = train_smp_lora.parse_args()

    assert args.optimizer == "adam"
    assert args.sgd_momentum == 0.9


def test_parse_args_accepts_optimizer_choice(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_smp_lora.py",
            "--random_init",
            "--member_dir",
            "member",
            "--nonmember_dir",
            "nonmember",
            "--optimizer",
            "adamw",
        ],
    )

    args = train_smp_lora.parse_args()

    assert args.optimizer == "adamw"


def test_set_global_seed_makes_torch_draws_reproducible():
    train_smp_lora.set_global_seed(123)
    first = torch.rand(4)

    train_smp_lora.set_global_seed(123)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_apply_seed_only_seeds_when_explicit(monkeypatch):
    seen = []

    monkeypatch.setattr(train_smp_lora, "set_global_seed", lambda seed: seen.append(seed))

    assert train_smp_lora.apply_seed(None) is False
    assert train_smp_lora.apply_seed(321) is True
    assert seen == [321]


def test_recommended_num_workers_scales_with_cpu_count():
    assert train_smp_lora.recommended_num_workers(cpu_count=1) == 0
    assert train_smp_lora.recommended_num_workers(cpu_count=8) == 4
    assert train_smp_lora.recommended_num_workers(cpu_count=64) == 8


def test_build_dataloader_kwargs_enables_worker_prefetch():
    kwargs = train_smp_lora.build_dataloader_kwargs(
        batch_size=8,
        shuffle=True,
        num_workers=4,
    )

    assert kwargs["batch_size"] == 8
    assert kwargs["shuffle"] is True
    assert kwargs["num_workers"] == 4
    assert kwargs["pin_memory"] is True
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_build_dataloader_kwargs_skips_prefetch_without_workers():
    kwargs = train_smp_lora.build_dataloader_kwargs(
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    assert kwargs["num_workers"] == 0
    assert kwargs["pin_memory"] is True
    assert "persistent_workers" not in kwargs
    assert "prefetch_factor" not in kwargs


def test_build_runtime_config_enables_cuda_throughput_defaults():
    config = train_smp_lora.build_runtime_config(
        device="cuda",
        throughput_mode=True,
        amp_dtype="bf16",
        allow_tf32=True,
        cudnn_benchmark=True,
        non_blocking_transfers=True,
    )

    assert config["throughput_mode"] is True
    assert config["autocast_enabled"] is True
    assert config["autocast_dtype"] is torch.bfloat16
    assert config["allow_tf32"] is True
    assert config["cudnn_benchmark"] is True
    assert config["non_blocking_transfers"] is True
    assert config["matmul_precision"] == "high"


def test_build_runtime_config_disables_cuda_only_features_on_cpu():
    config = train_smp_lora.build_runtime_config(
        device="cpu",
        throughput_mode=True,
        amp_dtype="bf16",
        allow_tf32=True,
        cudnn_benchmark=True,
        non_blocking_transfers=True,
    )

    assert config["throughput_mode"] is False
    assert config["autocast_enabled"] is False
    assert config["autocast_dtype"] is None
    assert config["allow_tf32"] is False
    assert config["cudnn_benchmark"] is False
    assert config["non_blocking_transfers"] is False


def test_move_tensor_to_device_passes_non_blocking_flag():
    class FakeTensor:
        def __init__(self):
            self.calls = []

        def to(self, device, non_blocking=False):
            self.calls.append((device, non_blocking))
            return self

    tensor = FakeTensor()
    moved = train_smp_lora.move_tensor_to_device(
        tensor,
        device=torch.device("cuda"),
        non_blocking=True,
    )

    assert moved is tensor
    assert tensor.calls == [(torch.device("cuda"), True)]


def test_save_checkpoint_can_skip_training_log(tmp_path, monkeypatch):
    monkeypatch.setattr(
        smp_module,
        "lora_injection_summary",
        lambda model: {
            "num_lora_layers": 1,
            "total_lora_params": 2,
            "total_original_params": 8,
            "overall_compression_ratio": 4.0,
            "layers": [],
        },
    )
    monkeypatch.setattr(
        "diffaudit.defenses.lora_ddpm.get_lora_state_dict",
        lambda model: {"dummy": torch.tensor([1.0])},
    )

    trainer = SMPLoRATrainer.__new__(SMPLoRATrainer)
    trainer.model = nn.Linear(1, 1)
    trainer.proxy_model = nn.Linear(1, 1)
    trainer.log = [{"step": 1, "objective": 0.1}]
    trainer.method = "smp"
    trainer.lambda_coeff = 0.1
    trainer.delta = 1e-4
    trainer.scheduler = SimpleNamespace(
        config={"num_train_timesteps": 1000, "beta_schedule": "linear"}
    )

    trainer.save_checkpoint(tmp_path / "step_100", include_training_log=False)
    trainer.save_checkpoint(tmp_path / "final", include_training_log=True)

    assert not (tmp_path / "step_100" / "training_log.json").exists()
    assert (tmp_path / "final" / "training_log.json").exists()


def test_smp_lora_trainer_uses_requested_adamw_optimizer(monkeypatch):
    monkeypatch.setattr(smp_module, "inject_lora_into_unet", lambda *args, **kwargs: {})
    monkeypatch.setattr(smp_module, "get_lora_parameters", lambda model: [model.weight])

    trainer = SMPLoRATrainer(
        model=nn.Linear(1, 1),
        optimizer="adamw",
        device="cpu",
    )

    assert isinstance(trainer.lora_optimizer, torch.optim.AdamW)
    assert isinstance(trainer.proxy_optimizer, torch.optim.AdamW)


def test_smp_lora_trainer_uses_requested_sgd_optimizer(monkeypatch):
    monkeypatch.setattr(smp_module, "inject_lora_into_unet", lambda *args, **kwargs: {})
    monkeypatch.setattr(smp_module, "get_lora_parameters", lambda model: [model.weight])

    trainer = SMPLoRATrainer(
        model=nn.Linear(1, 1),
        optimizer="sgd",
        sgd_momentum=0.9,
        device="cpu",
    )

    assert isinstance(trainer.lora_optimizer, torch.optim.SGD)
    assert isinstance(trainer.proxy_optimizer, torch.optim.SGD)
    assert trainer.lora_optimizer.param_groups[0]["momentum"] == 0.9
