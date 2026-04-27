# 2026-04-16 TMIA-DM GPU Pilot Blocker

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `TMIA-DM late-window GPU pilot`
- `selected_family`: `TMIA-DM late-window long_window`
- `attempted_device`: `cuda:0`
- `decision`: `blocked by local torch runtime`

## Question

After `TMIA-DM late-window long_window` became GPU-eligible on bounded CPU evidence, can the current local execution environment actually launch the first `GPU128` pilot?

## Executed Evidence

Attempted GPU pilot target:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/tmiadm-cifar10-late-window-protocol-probe-20260416-gpu-128-r1`

Environment checks:

- default `python`:
  - executable: `<PYTHON_EXE>`
  - `torch = 2.11.0+cpu`
  - `torch.version.cuda = None`
  - `torch.cuda.is_available() = False`
- alternate `miniforge` python:
  - executable: `<CONDA_PYTHON>`
  - current state: `ModuleNotFoundError: No module named 'torch'`

Observed host state:

- `nvidia-smi` still sees the physical GPU
- the blocker is therefore not “no NVIDIA device”
- it is “the current research Python runtime is not CUDA-capable”

## Verdict

Current verdict:

- `blocked by local torch runtime`

Reason:

1. the attempted `cuda:0` launch failed before real model execution because `torch.cuda.is_available()` is false;
2. the default interpreter is a CPU-only torch build;
3. the alternate interpreter currently lacks torch entirely;
4. therefore the next gate is environment-level, not method-level.

## Decision

Current release decision:

- `keep TMIA-DM as gpu-eligible in research logic`
- `mark local GPU execution as blocked`
- `do not claim a launched GPU rung`

Meaning:

1. the method has earned a GPU pilot on evidence grounds;
2. the local machine has not yet executed that pilot;
3. the blocker should be treated as execution-infra debt, not as a negative research verdict.

## Next Gate

The next unblock step should be one of:

1. provide a CUDA-capable torch environment for `Research`;
2. expose the exact approved interpreter path that already has CUDA support;
3. if neither is available, keep progressing CPU-side refinement while recording GPU as blocked.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: if shared execution environments are being standardized, this is a concrete handoff item: `Research` needs one admitted CUDA-capable Python runtime.
- Materials: wording should say `TMIA-DM late-window` is evidence-wise GPU-eligible, but the local GPU pilot is currently blocked by environment, not by a failed attack result.
