# pia_next_run

Minimal intake gate for the gray-box PIA next run.

It does not run the attack. It validates the declared next-run inputs and emits:

- `manifest.json`
- `provenance.json`

Current inputs:

- `--config`
- `--member-split-root`
- `--repo-root`
- `--out-dir`

PowerShell example:

```powershell
cd Research/tools/pia_next_run
.\run.ps1 --config ..\..\tmp\configs\pia-cifar10-graybox-assets.local.yaml --member-split-root ..\..\external\PIA\DDPM --repo-root ..\..\external\PIA --out-dir ..\..\tmp\pia_next_run_smoke
```

Exit codes:

- `0`: validation OK
- `2`: validation failed
