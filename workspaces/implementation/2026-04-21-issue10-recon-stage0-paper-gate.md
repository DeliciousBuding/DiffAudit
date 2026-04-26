# 2026-04-21 Issue #10: Recon Stage 0 Paper Gate

## Hypothesis

Strict `Attack-I` should not start from the current public `recon` bundle unless Stage 0 can prove paper-aligned target/shadow/member/non-member semantics. The existing local bundle audit is useful, but `local-semantic-chain-ready` is not equivalent to paper-aligned.

## Method

Added a dedicated Stage 0 gate:

```powershell
python -m diffaudit check-recon-stage0-paper-gate `
  --repo-root external/Reconstruction-based-Attack `
  --bundle-root D:/Code/DiffAudit/Download/black-box/supplementary/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models `
  --attack-scenario attack-i
```

The new command checks:

- upstream recon workspace exists and contains the required scripts
- public bundle exists
- public bundle remains locally self-consistent through `audit-recon-public-bundle`
- strict `paper_aligned_semantics` is true before allowing a paper-faithful start

## Evidence

Code / test anchors:

- `src/diffaudit/attacks/recon.py`: `check_recon_stage0_paper_gate`
- `src/diffaudit/cli.py`: `check-recon-stage0-paper-gate`
- `tests/test_recon_attack.py`: `test_cli_blocks_recon_stage0_when_bundle_is_proxy_semantic`

Validation run:

```powershell
python -m unittest tests.test_recon_attack -q
```

Result:

```text
Ran 25 tests in 68.897s
OK
```

`pytest` was not used because the default Python environment does not have `pytest` installed.

## Verdict

`positive hardening`.

Issue #10 is resolved as an executable gate, not by promoting the bundle. The current public `recon` bundle remains:

- `local_semantic_chain_ready = true` when the bundle audit passes
- `paper_aligned_semantics = false`
- `semantic_gate.current_state = proxy-shadow-member`
- `allowed_claim = local-semantic-chain-ready`

Therefore strict paper-faithful `Attack-I` remains `blocked` until better asset evidence appears.

## Handoff

No Runtime or Platform schema change is required. Downstream consumers should keep reading admitted `recon` as black-box risk evidence under `fine-tuned / controlled / public-subset / proxy-shadow-member`, not as a paper-aligned benchmark.

## Next Action

Keep black-box parked unless a real asset/boundary shift appears. Do not spend GPU on recon reruns for this issue.

