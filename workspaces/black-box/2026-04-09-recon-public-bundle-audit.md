# 2026-04-09 Black-Box Audit: Recon Public Bundle

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_track`: `recon`
- `current_state`: `real public bundle audited with the local semantic gate command`
- `evidence_level`: `asset-audit`

## A. Command Run

```powershell
conda run -n diffaudit-research python -m diffaudit audit-recon-public-bundle `
  --bundle-root external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models
```

## B. Audit Result

Current command returned:

- `status = ready`
- `semantic_gate.current_state = proxy-shadow-member`
- `semantic_gate.paper_aligned = false`
- `semantic_gate.allowed_claim = local-semantic-chain-ready`

Source datasets confirmed:

- `target_member`
  - exists, valid dict payload, `sample_count = 100`
- `target_non_member`
  - exists, valid dict payload, `sample_count = 100`
- `shadow_member_proxy`
  - exists, valid dict payload, `sample_count = 100`
- `shadow_non_member`
  - exists, valid dict payload, `sample_count = 100`

Derived variants confirmed:

- `derived-public-10`
- `derived-public-25`
- `derived-public-50`
- `derived-public-100`

All four variants currently have:

- `mapping_note_exists = true`
- `mapping_lines_complete = true`
- `shadow_member_is_proxy = true`

## C. Interpretation

This audit upgrades the black-box wording from:

- `we think the current mapping is locally consistent`

to:

- `the current public bundle is locally self-consistent and machine-audited`

But it does **not** upgrade the line to:

- `paper-aligned`

because the same audit also confirms that every currently derived public subset still depends on `shadow_member_proxy`.

So the strongest current claim is:

- `recon` now has a machine-audited local semantic chain
- the local chain is stable enough for admitted black-box evidence
- the semantic gate remains constrained by `proxy-shadow-member`

## D. Immediate Next Step

1. Keep `recon DDIM public-100 step30` as main evidence.
2. Keep the semantic caveat visible in all black-box summary docs.
3. Do not spend GPU on recon reruns until new paper-aligned split evidence appears.
