import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_mofit_interface_canary.py"
    spec = importlib.util.spec_from_file_location("run_mofit_interface_canary", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MoFitInterfaceCanaryTests(unittest.TestCase):
    def test_apply_launch_profile_tightens_cpu_first_budget(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            member_limit=3,
            nonmember_limit=4,
            member_offset=0,
            nonmember_offset=0,
            surrogate_steps=8,
            embedding_steps=12,
            surrogate_lr=1e-2,
            embedding_lr=5e-3,
            guidance_scale=3.5,
            max_timestep=140,
            device="cuda",
            launch_profile="bounded-cpu-first",
        )

        effective = module.apply_launch_profile(args)

        self.assertEqual(effective.launch_profile, "bounded-cpu-first")
        self.assertEqual(effective.member_limit, 1)
        self.assertEqual(effective.nonmember_limit, 1)
        self.assertEqual(effective.surrogate_steps, 1)
        self.assertEqual(effective.embedding_steps, 2)
        self.assertEqual(effective.device, "cpu")

    def test_apply_launch_profile_sets_cpu_micro_rung_budget(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            member_limit=8,
            nonmember_limit=8,
            member_offset=0,
            nonmember_offset=0,
            surrogate_steps=99,
            embedding_steps=99,
            surrogate_lr=1e-2,
            embedding_lr=5e-3,
            guidance_scale=3.5,
            max_timestep=140,
            device="cuda",
            launch_profile="cpu-micro-rung",
        )

        effective = module.apply_launch_profile(args)

        self.assertEqual(effective.launch_profile, "cpu-micro-rung")
        self.assertEqual(effective.member_limit, 2)
        self.assertEqual(effective.nonmember_limit, 2)
        self.assertEqual(effective.surrogate_steps, 2)
        self.assertEqual(effective.embedding_steps, 4)
        self.assertEqual(effective.device, "cpu")

    def test_apply_launch_profile_raises_defaults_to_cpu_micro_rung_budget(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            member_limit=1,
            nonmember_limit=1,
            member_offset=0,
            nonmember_offset=0,
            surrogate_steps=1,
            embedding_steps=2,
            surrogate_lr=1e-2,
            embedding_lr=5e-3,
            guidance_scale=3.5,
            max_timestep=140,
            device="cpu",
            launch_profile="cpu-micro-rung",
        )

        effective = module.apply_launch_profile(args)

        self.assertEqual(effective.member_limit, 2)
        self.assertEqual(effective.nonmember_limit, 2)
        self.assertEqual(effective.surrogate_steps, 2)
        self.assertEqual(effective.embedding_steps, 4)

    def test_apply_launch_profile_sets_cpu_review_rung_budget(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            member_limit=99,
            nonmember_limit=99,
            member_offset=0,
            nonmember_offset=0,
            surrogate_steps=99,
            embedding_steps=99,
            surrogate_lr=1e-2,
            embedding_lr=5e-3,
            guidance_scale=3.5,
            max_timestep=140,
            device="cuda",
            launch_profile="cpu-review-rung",
        )

        effective = module.apply_launch_profile(args)

        self.assertEqual(effective.launch_profile, "cpu-review-rung")
        self.assertEqual(effective.member_limit, 2)
        self.assertEqual(effective.nonmember_limit, 2)
        self.assertEqual(effective.surrogate_steps, 3)
        self.assertEqual(effective.embedding_steps, 6)
        self.assertEqual(effective.device, "cpu")

    def test_prepare_runner_args_adds_missing_resolution_default(self) -> None:
        module = _load_script_module()

        args = argparse.Namespace(
            member_limit=1,
            nonmember_limit=1,
            member_offset=0,
            nonmember_offset=0,
            surrogate_steps=1,
            embedding_steps=2,
            surrogate_lr=1e-2,
            embedding_lr=5e-3,
            guidance_scale=3.5,
            max_timestep=140,
            device="cpu",
            launch_profile="bounded-cpu-first",
        )

        prepared = module.prepare_runner_args(args)

        self.assertEqual(prepared.resolution, 512)

    def test_materialize_tensor_converts_inference_tensor_for_backward_use(self) -> None:
        module = _load_script_module()

        with torch.inference_mode():
            inference_tensor = torch.ones((1, 2), dtype=torch.float32)

        materialized = module.materialize_tensor(inference_tensor)
        weight = torch.nn.Parameter(torch.ones((2, 1), dtype=torch.float32))
        loss = (materialized @ weight).sum()
        loss.backward()

        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.allclose(weight.grad, torch.ones_like(weight.grad)))

    def test_load_rows_honors_offset_and_limit(self) -> None:
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "member"
            split_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = split_dir / "metadata.jsonl"
            metadata_path.write_text(
                "\n".join(
                    [
                        json.dumps({"file_name": "a.png", "text": "a"}),
                        json.dumps({"file_name": "b.png", "text": "b"}),
                        json.dumps({"file_name": "c.png", "text": "c"}),
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.load_rows(split_dir, subset_limit=1, offset=1)

            self.assertEqual(rows, [{"file_name": "b.png", "text": "b"}])

    def test_run_canary_executes_bounded_rows_and_updates_summary(self) -> None:
        module = _load_script_module()
        from diffaudit.attacks.mofit_scaffold import execute_mofit_sample

        class FakeRunner:
            def execute_row(self, *, split_name: str, split_dir: Path, row: dict, run_root: Path, args) -> dict:
                del split_dir

                def unet_forward(
                    latent_value: torch.Tensor,
                    timestep_value: torch.Tensor,
                    embedding_value: torch.Tensor,
                ) -> SimpleNamespace:
                    del timestep_value
                    embed_scalar = embedding_value.mean(dim=-1, keepdim=True).to(dtype=latent_value.dtype)
                    return SimpleNamespace(sample=latent_value + embed_scalar)

                return execute_mofit_sample(
                    run_root=run_root,
                    split=split_name,
                    row=row,
                    latent=torch.tensor([[1.0]], dtype=torch.float32),
                    timestep=args.max_timestep,
                    cond_embedding=torch.tensor([[2.0]], dtype=torch.float32),
                    uncond_embedding=torch.tensor([[0.5]], dtype=torch.float32),
                    unet_forward_fn=unet_forward,
                    guidance_scale=1.0,
                    surrogate_steps=args.surrogate_steps,
                    surrogate_lr=0.5,
                    embedding_steps=args.embedding_steps,
                    embedding_lr=0.1,
                    initial_surrogate_latent=torch.tensor([[0.0]], dtype=torch.float32),
                    initial_embedding=torch.tensor([[0.0]], dtype=torch.float32),
                    caption_fallback_fn=lambda current_row: f"fallback:{current_row['file_name']}",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            member_dir = root / "member"
            nonmember_dir = root / "nonmember"
            member_dir.mkdir(parents=True, exist_ok=True)
            nonmember_dir.mkdir(parents=True, exist_ok=True)
            (member_dir / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"file_name": "m0.png", "text": "skip member"}),
                        json.dumps({"file_name": "m1.png", "text": "member prompt"}),
                        json.dumps({"file_name": "m2.png", "text": "member prompt 2"}),
                    ]
                ),
                encoding="utf-8",
            )
            (nonmember_dir / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"file_name": "n0.png", "text": "skip nonmember"}),
                        json.dumps({"file_name": "n1.png", "text": "   "}),
                        json.dumps({"file_name": "n2.png", "text": "nonmember prompt 2"}),
                    ]
                ),
                encoding="utf-8",
            )

            args = argparse.Namespace(
                run_root=root / "run",
                member_dir=member_dir,
                nonmember_dir=nonmember_dir,
                model_dir=root / "model",
                lora_dir=root / "lora",
                blip_dir=root / "blip",
                member_limit=8,
                member_offset=1,
                nonmember_limit=8,
                nonmember_offset=1,
                device="cpu",
                surrogate_steps=2,
                embedding_steps=4,
                surrogate_lr=0.5,
                embedding_lr=0.1,
                guidance_scale=1.0,
                max_timestep=10,
                record_trace=True,
                launch_profile="cpu-review-rung",
            )

            payload = module.run_canary(args, runner=FakeRunner())

            self.assertEqual(payload["status"], "canary_executed")
            self.assertEqual(payload["launch_profile"], "cpu-review-rung")
            self.assertEqual(payload["executed_member_count"], 2)
            self.assertEqual(payload["executed_nonmember_count"], 2)
            self.assertEqual(payload["member_rows"][0]["file_name"], "m1.png")
            self.assertEqual(payload["nonmember_rows"][0]["file_name"], "n1.png")

            summary = json.loads((args.run_root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "canary_executed")
            self.assertEqual(summary["launch_profile"], "cpu-review-rung")
            self.assertEqual(summary["surrogate_steps"], 3)
            self.assertEqual(summary["embedding_steps"], 6)
            self.assertEqual(summary["device"], "cpu")
            self.assertEqual(summary["executed_member_count"], 2)
            self.assertEqual(summary["executed_nonmember_count"], 2)

            records = [
                json.loads(line)
                for line in (args.run_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(records), 4)
            self.assertEqual(records[0]["file_name"], "m1.png")
            self.assertEqual(records[2]["prompt_source"], "blip-fallback")

            surrogate_trace = json.loads((args.run_root / "traces" / "surrogate" / "member-m1.json").read_text(encoding="utf-8"))
            embedding_trace = json.loads((args.run_root / "traces" / "embedding" / "member-m1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(surrogate_trace), 3)
            self.assertEqual(len(embedding_trace), 6)


if __name__ == "__main__":
    unittest.main()
