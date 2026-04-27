# gsa_next_run (GSA intake gate)

This directory is self-contained and only depends on Python (3.10+). It provides a runnable CLI that:

- validates `assets_root` / `repo_root` / `config`
- hashes input files under `assets_root`
- hashes config content (file or directory)
- captures git commit info (if `repo_root` is a git work tree)
- writes `manifest.json` and `provenance.json`

## Quickstart (no install)

From this directory:

```powershell
.\examples\minimal\run_example.ps1
```

Exit code is `0` on success, `2` on validation failure.

## Example details (tiny git repo)

The sample auto-initializes a tiny git repo inside `examples/minimal/repo_root` so the gate can record a commit.

```powershell
cd <DIFFAUDIT_ROOT>\Research\tools\gsa_next_run
.\examples\minimal\run_example.ps1
```

Expected key lines (illustrative):

- `OK=true manifest=...\\manifest.json provenance=...\\provenance.json`
- `WARN repo_root has uncommitted changes ...` (only if repo is dirty and `--strict` is not used)

## Install (optional)

```powershell
pip install -e .
gsa-next-run --help
```

## Output files

`manifest.json` contains input/config hashes and git info:

- `inputs.assets.tree_sha256` and per-file sha256 list
- `inputs.config_sha256` (file hash or directory tree hash)
- `git.commit` (if available)
- `validation.ok/errors/warnings`

`provenance.json` contains runtime metadata:

- `manifest_sha256` (hash of canonicalized manifest)
- `host` (hostname, uname, python, pid, cwd)
- `created_at` timestamp with timezone offset

## CLI reference

```powershell
.\run.ps1 --help
```

Flags:

- `--assets-root <dir>`
- `--repo-root <dir>`
- `--config <file|dir>`
- `--out-dir <dir>` (default `.`)
- `--strict` (require git + clean repo)
- `--stdout` (emit JSON to stdout instead of writing files)
