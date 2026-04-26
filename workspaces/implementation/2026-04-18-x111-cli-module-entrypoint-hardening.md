# 2026-04-18 X-111 CLI Module Entrypoint Hardening

## Question

Can `Research` expose one honest module-level CLI entrypoint so `python -m diffaudit.cli ...` actually executes `main()` instead of silently returning?

## Inputs Reviewed

- `D:\Code\DiffAudit\Research\src\diffaudit\cli.py`
- `D:\Code\DiffAudit\Research\src\diffaudit\__main__.py`
- `D:\Code\DiffAudit\Research\tests\test_risk_targeted_unlearning.py`
- `D:\Code\DiffAudit\Runtime-Server\runners\gsa-runner\run.py`
- `D:\Code\DiffAudit\Runtime-Server\runners\pia-runner\run.py`
- `D:\Code\DiffAudit\Runtime-Server\runners\recon-runner\run.py`

## What Landed

### 1. One real regression test now exists

The repo now contains a direct subprocess-level test:

- `D:\Code\DiffAudit\Research\tests\test_cli_module_entrypoint.py`

It verifies that:

1. `python -m diffaudit.cli unsupported-command`
2. returns non-zero
3. and actually emits CLI parser stderr

Before this fix, that module invocation exited `0` and produced no parser output, which means `main()` was never reached.

### 2. The module entrypoint is now hardened

`D:\Code\DiffAudit\Research\src\diffaudit\cli.py` now explicitly ends with:

- `if __name__ == "__main__":`
- `raise SystemExit(main())`

So module invocation and direct function import now share the same execution path.

## Actual Read

This is not a new research result.

It is runtime-interface hardening:

1. it removes one silent failure mode from the `Research` execution surface
2. it makes ad-hoc local runs consistent with the existing Runtime runner pattern
3. it reduces friction for future `Research -> Runtime -> Platform` orchestration, because module invocation now behaves like a normal CLI

## Verdict

- `x111_cli_module_entrypoint_verdict = positive hardening`

## Verification

Verified with:

- `conda run -n diffaudit-research python -m unittest tests.test_cli_module_entrypoint tests.test_risk_targeted_unlearning`

Result:

- `11 tests`
- `OK`

## Canonical Evidence Anchor

- `D:\Code\DiffAudit\Research\workspaces\implementation\2026-04-18-x111-cli-module-entrypoint-hardening.md`

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `Platform/Runtime`: no schema change required
- `Runtime runners`: already compatible, but future module-level invocation is now safe
