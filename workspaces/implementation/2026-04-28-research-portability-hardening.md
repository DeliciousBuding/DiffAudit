# 2026-04-28 Research Portability Hardening

## Question

Can a new teammate clone `Research/`, understand where raw assets belong, and continue work without inheriting one author's absolute repo-root or home-directory assumptions from tracked docs, scripts, or evidence files?

## Method

Performed one tracked-text portability sweep across active and historical surfaces:

- `docs/`
- `workspaces/`
- `experiments/`
- `legacy/`
- selected tests and code defaults

Applied a single portability policy:

- cross-repo and raw-asset references use `<DIFFAUDIT_ROOT>/...`
- repo-local code defaults derive from the checked-out repo root when execution semantics matter
- machine-user paths are replaced by generic placeholders or generic user examples
- operator-local Feishu publishing rules are kept outside the portable `Research` contract

## Key Fixes

### 1. Teammate-facing docs no longer assume the author's machine

- active onboarding and storage docs now point teammates to `<DIFFAUDIT_ROOT>/Research` and `<DIFFAUDIT_ROOT>/Download`
- `docs/paper-reports/README.md` and `docs/paper-reports/report-spec.md` no longer require legacy `LocalOps/feishu/...` paths as if they were portable prerequisites
- `legacy/local_api/README.md` now maps the live control plane to `<DIFFAUDIT_ROOT>/Runtime-Server` and treats `Services/Local-API` as a historical name

### 2. Historical evidence is sanitized instead of leaking author paths

- tracked `workspaces/runtime/jobs/*.json`
- tracked `experiments/*/summary.json`
- tracked workspace verdict notes and audit packets

These now use placeholders such as `<DIFFAUDIT_ROOT>`, `<RESEARCH_PYTHON>`, and `<PYTHON_EXE>` instead of author-machine absolute paths.

### 3. Code-level defaults were hardened, not just rewritten

- `src/diffaudit/defenses/risk_targeted_unlearning.py` now derives the default checkpoint root from the repository layout instead of baking in one machine path
- `scripts/prepare_clid_sanitized_probe.py` now rewrites the fallback finalize template via a regex that tolerates either the old absolute path or the new portable form
- portability tests now check for generic hardcoded-author patterns rather than one specific username string

## Evidence

Path scan result:

- searched tracked text for author-machine absolute repo roots and author-home path leaks
- result: no remaining tracked-text hits in `Research/` outside the excluded handoff transcript

Validation:

```powershell
python -m py_compile scripts/prepare_clid_sanitized_probe.py src/diffaudit/defenses/risk_targeted_unlearning.py
conda run -n diffaudit-research python -m pytest tests/test_risk_targeted_unlearning.py tests/test_dpdm_launch_portability.py tests/test_clid_threshold_audit.py tests/test_validate_intake_index.py tests/test_validate_attack_defense_table.py -q
```

Result:

```text
29 passed in 17.40s
```

## Verdict

`positive hardening`.

`Research` is now materially more portable for teammate takeover:

- raw assets stay conceptually outside the repo under `Download/`
- onboarding/docs no longer depend on `D:` or one Windows username
- executable code defaults no longer hide author-machine assumptions
- historical evidence can still be read, but no longer leaks local absolute paths

## GPU State

- `active_gpu_question = none`
- `next_gpu_candidate = none`
- current work stayed CPU-only

## Handoff

No Platform or Runtime schema change is required.

If a downstream consumer wants to reference the cleanup:

- use [docs/storage-boundary.md](../../docs/storage-boundary.md) for the canonical storage rule
- use [docs/data-and-assets-handoff.md](../../docs/data-and-assets-handoff.md) and [docs/teammate-setup.md](../../docs/teammate-setup.md) for teammate takeover
- use this note as the canonical portability-hardening verdict
