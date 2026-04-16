import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch


class MoFitScaffoldTests(unittest.TestCase):
    def test_initialize_mofit_scaffold_creates_expected_artifacts(self) -> None:
        from diffaudit.attacks.mofit_scaffold import initialize_mofit_scaffold

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            result = initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=8,
                embedding_steps=12,
                member_count=1,
                nonmember_count=1,
            )

            summary_path = run_root / "summary.json"
            records_path = run_root / "records.jsonl"
            surrogate_trace_dir = run_root / "traces" / "surrogate"
            embedding_trace_dir = run_root / "traces" / "embedding"

            self.assertEqual(result["summary_path"], summary_path.as_posix())
            self.assertTrue(summary_path.exists())
            self.assertTrue(records_path.exists())
            self.assertTrue(surrogate_trace_dir.exists())
            self.assertTrue(embedding_trace_dir.exists())
            self.assertEqual(records_path.read_text(encoding="utf-8"), "")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_schema_version"], "mofit-interface-canary.v1")
            self.assertEqual(summary["status"], "scaffold_only")
            self.assertEqual(summary["target_family"], "sd15-plus-celeba_partial_target-checkpoint-25000")
            self.assertEqual(summary["caption_source"], "metadata-with-blip-fallback")
            self.assertEqual(summary["surrogate_steps"], 8)
            self.assertEqual(summary["embedding_steps"], 12)
            self.assertEqual(summary["member_count"], 1)
            self.assertEqual(summary["nonmember_count"], 1)

    def test_append_mofit_record_creates_trace_placeholders(self) -> None:
        from diffaudit.attacks.mofit_scaffold import append_mofit_record, initialize_mofit_scaffold

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=8,
                embedding_steps=12,
                member_count=1,
                nonmember_count=1,
            )

            record = append_mofit_record(
                run_root=run_root,
                split="member",
                file_name="sample.png",
                prompt_source="metadata",
                prompt_text="face portrait",
            )

            records_path = run_root / "records.jsonl"
            lines = records_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)

            row = json.loads(lines[0])
            self.assertEqual(row["split"], "member")
            self.assertEqual(row["file_name"], "sample.png")
            self.assertEqual(row["prompt_source"], "metadata")
            self.assertEqual(row["prompt_text"], "face portrait")
            self.assertIsNone(row["l_cond"])
            self.assertIsNone(row["l_uncond"])
            self.assertIsNone(row["mofit_score"])

            surrogate_trace = Path(row["surrogate_trace_path"])
            embedding_trace = Path(row["embedding_trace_path"])
            self.assertTrue(surrogate_trace.exists())
            self.assertTrue(embedding_trace.exists())
            self.assertEqual(json.loads(surrogate_trace.read_text(encoding="utf-8")), [])
            self.assertEqual(json.loads(embedding_trace.read_text(encoding="utf-8")), [])

            self.assertEqual(record["surrogate_trace_path"], surrogate_trace.as_posix())
            self.assertEqual(record["embedding_trace_path"], embedding_trace.as_posix())

    def test_finalize_mofit_record_updates_scores_and_traces(self) -> None:
        from diffaudit.attacks.mofit_scaffold import (
            append_mofit_record,
            finalize_mofit_record,
            initialize_mofit_scaffold,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=8,
                embedding_steps=12,
                member_count=1,
                nonmember_count=1,
            )
            record = append_mofit_record(
                run_root=run_root,
                split="member",
                file_name="sample.png",
                prompt_source="metadata",
                prompt_text="face portrait",
            )

            finalized = finalize_mofit_record(
                run_root=run_root,
                file_name="sample.png",
                split="member",
                l_cond=1.25,
                l_uncond=0.75,
                mofit_score=0.5,
                surrogate_trace=[{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.8}],
                embedding_trace=[{"step": 0, "loss": 0.6}, {"step": 1, "loss": 0.4}],
            )

            self.assertEqual(finalized["l_cond"], 1.25)
            self.assertEqual(finalized["l_uncond"], 0.75)
            self.assertEqual(finalized["mofit_score"], 0.5)

            lines = (run_root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0])
            self.assertEqual(row["l_cond"], 1.25)
            self.assertEqual(row["l_uncond"], 0.75)
            self.assertEqual(row["mofit_score"], 0.5)

            surrogate_trace = Path(record["surrogate_trace_path"])
            embedding_trace = Path(record["embedding_trace_path"])
            self.assertEqual(
                json.loads(surrogate_trace.read_text(encoding="utf-8")),
                [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.8}],
            )
            self.assertEqual(
                json.loads(embedding_trace.read_text(encoding="utf-8")),
                [{"step": 0, "loss": 0.6}, {"step": 1, "loss": 0.4}],
            )

    def test_run_surrogate_optimization_reduces_toy_loss(self) -> None:
        from diffaudit.attacks.mofit_scaffold import run_surrogate_optimization

        initial = torch.tensor([2.0], dtype=torch.float32)

        def loss_fn(current: torch.Tensor) -> torch.Tensor:
            return ((current - 0.5) ** 2).mean()

        optimized, trace = run_surrogate_optimization(
            initial_value=initial,
            loss_fn=loss_fn,
            steps=4,
            lr=0.25,
        )

        self.assertEqual(len(trace), 4)
        self.assertLess(trace[-1]["loss"], trace[0]["loss"])
        self.assertLess(abs(float(optimized.item()) - 0.5), abs(float(initial.item()) - 0.5))

    def test_run_embedding_optimization_reduces_toy_loss(self) -> None:
        from diffaudit.attacks.mofit_scaffold import run_embedding_optimization

        initial = torch.tensor([1.5], dtype=torch.float32)

        def loss_fn(current: torch.Tensor) -> torch.Tensor:
            return ((current + 0.25) ** 2).mean()

        optimized, trace = run_embedding_optimization(
            initial_value=initial,
            loss_fn=loss_fn,
            steps=5,
            lr=0.2,
        )

        self.assertEqual(len(trace), 5)
        self.assertLess(trace[-1]["loss"], trace[0]["loss"])
        self.assertLess(abs(float(optimized.item()) + 0.25), abs(float(initial.item()) + 0.25))

    def test_compute_mofit_loss_terms_returns_cond_uncond_and_score(self) -> None:
        from diffaudit.attacks.mofit_scaffold import compute_mofit_loss_terms

        latent = torch.tensor([[1.0]], dtype=torch.float32)
        target_noise = torch.tensor([[0.0]], dtype=torch.float32)
        cond_embed = torch.tensor([[2.0]], dtype=torch.float32)
        uncond_embed = torch.tensor([[0.5]], dtype=torch.float32)

        def predict_noise_fn(
            latent_value: torch.Tensor,
            timestep: int,
            embed_value: torch.Tensor,
        ) -> torch.Tensor:
            return latent_value + embed_value + float(timestep) * 0.0

        losses = compute_mofit_loss_terms(
            latent=latent,
            timestep=10,
            target_noise=target_noise,
            cond_embedding=cond_embed,
            uncond_embedding=uncond_embed,
            predict_noise_fn=predict_noise_fn,
        )

        self.assertAlmostEqual(losses["l_cond"], 9.0)
        self.assertAlmostEqual(losses["l_uncond"], 2.25)
        self.assertAlmostEqual(losses["mofit_score"], 6.75)

    def test_build_mofit_loss_functions_work_with_optimization_helpers(self) -> None:
        from diffaudit.attacks.mofit_scaffold import (
            build_embedding_loss_fn,
            build_surrogate_loss_fn,
            run_embedding_optimization,
            run_surrogate_optimization,
        )

        target_noise = torch.tensor([[0.0]], dtype=torch.float32)
        timestep = 10
        uncond_embedding = torch.tensor([[0.0]], dtype=torch.float32)

        def predict_noise_fn(
            latent_value: torch.Tensor,
            timestep_value: int,
            embed_value: torch.Tensor,
        ) -> torch.Tensor:
            return latent_value + embed_value + float(timestep_value) * 0.0

        initial_surrogate = torch.tensor([[2.0]], dtype=torch.float32)
        surrogate_loss_fn = build_surrogate_loss_fn(
            timestep=timestep,
            target_noise=target_noise,
            uncond_embedding=uncond_embedding,
            predict_noise_fn=predict_noise_fn,
        )
        optimized_surrogate, surrogate_trace = run_surrogate_optimization(
            initial_value=initial_surrogate,
            loss_fn=surrogate_loss_fn,
            steps=4,
            lr=0.25,
        )
        self.assertLess(surrogate_trace[-1]["loss"], surrogate_trace[0]["loss"])

        initial_embedding = torch.tensor([[1.5]], dtype=torch.float32)
        embedding_loss_fn = build_embedding_loss_fn(
            latent=optimized_surrogate,
            timestep=timestep,
            target_noise=target_noise,
            predict_noise_fn=predict_noise_fn,
        )
        optimized_embedding, embedding_trace = run_embedding_optimization(
            initial_value=initial_embedding,
            loss_fn=embedding_loss_fn,
            steps=5,
            lr=0.2,
        )
        self.assertLess(embedding_trace[-1]["loss"], embedding_trace[0]["loss"])
        self.assertLess(
            abs(float((optimized_surrogate + optimized_embedding).item())),
            abs(float((optimized_surrogate + initial_embedding).item())),
        )

    def test_build_unet_noise_predictor_adapts_unet_style_forward(self) -> None:
        from diffaudit.attacks.mofit_scaffold import build_unet_noise_predictor

        captured: dict[str, torch.Tensor] = {}

        def unet_forward(
            latent_value: torch.Tensor,
            timestep_value: torch.Tensor,
            embedding_value: torch.Tensor,
        ) -> SimpleNamespace:
            captured["latent_shape"] = torch.tensor(latent_value.shape)
            captured["timestep"] = timestep_value.detach().clone()
            captured["embedding_dtype"] = torch.tensor([1 if embedding_value.dtype == latent_value.dtype else 0])
            embed_scalar = embedding_value.mean(dim=-1, keepdim=True).to(dtype=latent_value.dtype)
            return SimpleNamespace(sample=latent_value + embed_scalar)

        predictor = build_unet_noise_predictor(unet_forward)
        latent = torch.tensor([[1.0]], dtype=torch.float32)
        embedding = torch.tensor([[2.0]], dtype=torch.float64)

        prediction = predictor(latent, 13, embedding)

        self.assertTrue(torch.equal(captured["latent_shape"], torch.tensor([1, 1])))
        self.assertEqual(captured["timestep"].dtype, torch.long)
        self.assertEqual(captured["timestep"].shape, torch.Size([1]))
        self.assertEqual(int(captured["timestep"].item()), 13)
        self.assertEqual(int(captured["embedding_dtype"].item()), 1)
        self.assertTrue(torch.allclose(prediction, torch.tensor([[3.0]], dtype=torch.float32)))

    def test_real_path_helpers_bridge_unet_style_predictor_and_target_noise(self) -> None:
        from diffaudit.attacks.mofit_scaffold import (
            build_embedding_loss_fn,
            build_surrogate_loss_fn,
            build_unet_noise_predictor,
            compute_guided_target_noise,
            compute_mofit_loss_terms,
        )

        def unet_forward(
            latent_value: torch.Tensor,
            timestep_value: torch.Tensor,
            embedding_value: torch.Tensor,
        ) -> SimpleNamespace:
            del timestep_value
            embed_scalar = embedding_value.mean(dim=-1, keepdim=True).to(dtype=latent_value.dtype)
            return SimpleNamespace(sample=latent_value + embed_scalar)

        predictor = build_unet_noise_predictor(unet_forward)
        latent = torch.tensor([[1.0]], dtype=torch.float32)
        cond_embedding = torch.tensor([[2.0]], dtype=torch.float32)
        uncond_embedding = torch.tensor([[0.5]], dtype=torch.float32)
        target_noise = compute_guided_target_noise(
            latent=latent,
            timestep=10,
            cond_embedding=cond_embedding,
            uncond_embedding=uncond_embedding,
            guidance_scale=1.0,
            predict_noise_fn=predictor,
        )

        losses = compute_mofit_loss_terms(
            latent=latent,
            timestep=10,
            target_noise=target_noise,
            cond_embedding=cond_embedding,
            uncond_embedding=uncond_embedding,
            predict_noise_fn=predictor,
        )

        self.assertAlmostEqual(float(target_noise.item()), 3.0)
        self.assertAlmostEqual(losses["l_cond"], 0.0)
        self.assertAlmostEqual(losses["l_uncond"], 2.25)
        self.assertAlmostEqual(losses["mofit_score"], -2.25)

        surrogate_loss_fn = build_surrogate_loss_fn(
            timestep=10,
            target_noise=target_noise,
            uncond_embedding=uncond_embedding,
            predict_noise_fn=predictor,
        )
        embedding_loss_fn = build_embedding_loss_fn(
            latent=latent,
            timestep=10,
            target_noise=target_noise,
            predict_noise_fn=predictor,
        )

        self.assertAlmostEqual(float(surrogate_loss_fn(latent).item()), 2.25)
        self.assertAlmostEqual(float(embedding_loss_fn(cond_embedding).item()), 0.0)

    def test_execute_mofit_sample_assembles_metadata_prompt_and_record_flow(self) -> None:
        from diffaudit.attacks.mofit_scaffold import execute_mofit_sample, initialize_mofit_scaffold

        def unet_forward(
            latent_value: torch.Tensor,
            timestep_value: torch.Tensor,
            embedding_value: torch.Tensor,
        ) -> SimpleNamespace:
            del timestep_value
            embed_scalar = embedding_value.mean(dim=-1, keepdim=True).to(dtype=latent_value.dtype)
            return SimpleNamespace(sample=latent_value + embed_scalar)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=5,
                embedding_steps=25,
                member_count=1,
                nonmember_count=1,
            )

            finalized = execute_mofit_sample(
                run_root=run_root,
                split="member",
                row={"file_name": "sample.png", "text": "face portrait"},
                latent=torch.tensor([[1.0]], dtype=torch.float32),
                timestep=10,
                cond_embedding=torch.tensor([[2.0]], dtype=torch.float32),
                uncond_embedding=torch.tensor([[0.5]], dtype=torch.float32),
                unet_forward_fn=unet_forward,
                guidance_scale=1.0,
                surrogate_steps=5,
                surrogate_lr=0.5,
                embedding_steps=25,
                embedding_lr=0.1,
                initial_surrogate_latent=torch.tensor([[0.0]], dtype=torch.float32),
                initial_embedding=torch.tensor([[0.0]], dtype=torch.float32),
            )

            self.assertEqual(finalized["split"], "member")
            self.assertEqual(finalized["file_name"], "sample.png")
            self.assertEqual(finalized["prompt_source"], "metadata")
            self.assertEqual(finalized["prompt_text"], "face portrait")
            self.assertLess(finalized["l_cond"], 0.1)
            self.assertLess(finalized["l_uncond"], 0.1)
            self.assertLess(abs(finalized["mofit_score"]), 0.1)

            surrogate_trace = json.loads(
                (run_root / "traces" / "surrogate" / "member-sample.json").read_text(encoding="utf-8")
            )
            embedding_trace = json.loads(
                (run_root / "traces" / "embedding" / "member-sample.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(surrogate_trace), 5)
            self.assertEqual(len(embedding_trace), 25)

    def test_execute_mofit_sample_uses_caption_fallback_when_metadata_missing(self) -> None:
        from diffaudit.attacks.mofit_scaffold import execute_mofit_sample, initialize_mofit_scaffold

        def unet_forward(
            latent_value: torch.Tensor,
            timestep_value: torch.Tensor,
            embedding_value: torch.Tensor,
        ) -> SimpleNamespace:
            del timestep_value
            embed_scalar = embedding_value.mean(dim=-1, keepdim=True).to(dtype=latent_value.dtype)
            return SimpleNamespace(sample=latent_value + embed_scalar)

        def caption_fallback(row: dict) -> str:
            self.assertEqual(row["file_name"], "fallback.png")
            return "blip caption"

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "mofit-scaffold"
            initialize_mofit_scaffold(
                run_root=run_root,
                target_family="sd15-plus-celeba_partial_target-checkpoint-25000",
                caption_source="metadata-with-blip-fallback",
                surrogate_steps=0,
                embedding_steps=0,
                member_count=1,
                nonmember_count=1,
            )

            finalized = execute_mofit_sample(
                run_root=run_root,
                split="nonmember",
                row={"file_name": "fallback.png", "text": "   "},
                latent=torch.tensor([[1.0]], dtype=torch.float32),
                timestep=10,
                cond_embedding=torch.tensor([[0.5]], dtype=torch.float32),
                uncond_embedding=torch.tensor([[0.5]], dtype=torch.float32),
                unet_forward_fn=unet_forward,
                guidance_scale=1.0,
                surrogate_steps=0,
                surrogate_lr=0.5,
                embedding_steps=0,
                embedding_lr=0.1,
                initial_surrogate_latent=torch.tensor([[1.0]], dtype=torch.float32),
                initial_embedding=torch.tensor([[0.5]], dtype=torch.float32),
                caption_fallback_fn=caption_fallback,
            )

            self.assertEqual(finalized["prompt_source"], "blip-fallback")
            self.assertEqual(finalized["prompt_text"], "blip caption")
            self.assertAlmostEqual(finalized["l_cond"], 0.0)
            self.assertAlmostEqual(finalized["l_uncond"], 0.0)
            self.assertAlmostEqual(finalized["mofit_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
