# 2026-04-16 Noise-as-a-Probe Defended-Extension Feasibility Review

## Question

After `Noise as a Probe` has become a strengthened bounded challenger candidate, is there one honest minimal defended-extension question that should become the next live task, or is defended extension still below release for this family?

## Inputs Reviewed

- `workspaces/gray-box/2026-04-16-noise-as-probe-protocol-asset-contract.md`
- `workspaces/gray-box/2026-04-16-noise-as-probe-larger-rung-repeat-verdict.md`
- `scripts/run_noise_as_probe_interface_canary.py`
- `src/diffaudit/attacks/pia_adapter.py`
- `src/diffaudit/attacks/tmiadm_adapter.py`
- `workspaces/gray-box/2026-04-16-tmiadm-temporal-striding-defense-verdict.md`
- `Download/shared/weights/stable-diffusion-v1-5/unet/config.json`

## Review

### 1. Direct `stochastic-dropout` port is not currently honest

`PIA` and `TMIA-DM` defense hooks are real because their adapters explicitly:

- expose `stochastic_dropout_defense`
- expose `dropout_activation_schedule`
- toggle `model.train()` / `model.eval()` at controlled timesteps

`Noise as a Probe` currently has none of that:

- the current script only implements:
  - prompt resolution
  - image encoding
  - DDIM inversion
  - target-model replay from injected latents
  - final distance scoring
- there is no defense schedule interface
- there is no timestep-wise train/eval switching hook

The stronger blocker is structural:

- the local `SD1.5` U-Net contains dropout modules
- but the loaded module scan shows:
  - `dropout_count = 70`
  - `p_values = [0.0]`

So a naive port of `stochastic-dropout(all_steps)` would not be a minimal defense toggle on the current contract. It would first require changing effective dropout probabilities or otherwise rewriting the execution surface.

### 2. `temporal-striding` does not transfer cleanly either

`TMIA-DM temporal-striding(stride=2)` is meaningful because that attack explicitly depends on a timestep window and can be weakened by exposing fewer late timesteps.

`Noise as a Probe` does not currently consume a timestep-series feature surface:

- it uses a fixed-step inversion path
- then a fixed-step replay path
- then one final image-distance score

Changing those timesteps would alter the attack contract itself, not simply add a small defense on top of the same scoring surface.

### 3. Current defended-extension reading

The current family therefore does **not** yet have:

- one minimal defense hook comparable to `PIA + stochastic-dropout`
- one challenger-specific defended mechanism comparable to `TMIA + temporal-striding`

Any future defense for this family would need a separate design note first, such as:

- replay-side stochasticity that is actually effective on this U-Net contract
- inversion/replay-step thinning with a clear argument that the attack definition is still comparable
- another latent-diffusion-specific perturbation that does not merely redefine the attack

That is a future design branch, not the next immediately releaseable task.

## Verdict

- `defended_extension_feasibility = no-go for now`
- `next_gpu_candidate = none`
- `direct_stochastic_dropout_port = rejected`
- `direct_temporal_striding_port = rejected`
- `next_step = switch away from defended execution and only reopen if a new defense-design note defines a truly bounded latent-diffusion-specific mechanism`

## Handoff Decision

- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

Reason:

- this changes internal task selection and prevents wasteful GPU work, but it does not change higher-layer packaging.
