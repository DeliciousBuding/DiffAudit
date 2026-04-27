# Research Storage Layout Finalization

Date: 2026-04-26

Verdict: `positive`

## Decision

The storage boundary cleanup is now physically aligned with the documented rule:

- `<DIFFAUDIT_ROOT>/Research/external/` is for upstream or exploratory code clones.
- `<DIFFAUDIT_ROOT>/Research/third_party/` is for minimal vendored integration code.
- `<DIFFAUDIT_ROOT>/Download/` is the raw intake layer for datasets, weights, supplementary bundles, manifests, and author release mirrors.
- `<DIFFAUDIT_ROOT>/Research/workspaces/<lane>/assets/` is for lane-normalized gateways and manifests.
- `<DIFFAUDIT_ROOT>/Research/workspaces/<lane>/runs/` is for evidence and run outputs.

## Physical Cleanup

Moved the raw intake tree out of the Research external-code surface:

- from `<DIFFAUDIT_ROOT>/Research/external/downloads/black-box/`
- from `<DIFFAUDIT_ROOT>/Research/external/downloads/gray-box/`
- from `<DIFFAUDIT_ROOT>/Research/external/downloads/manifests/`
- from `<DIFFAUDIT_ROOT>/Research/external/downloads/shared/`
- from `<DIFFAUDIT_ROOT>/Research/external/downloads/white-box/`

to:

- `<DIFFAUDIT_ROOT>/Download/black-box/`
- `<DIFFAUDIT_ROOT>/Download/gray-box/`
- `<DIFFAUDIT_ROOT>/Download/manifests/`
- `<DIFFAUDIT_ROOT>/Download/shared/`
- `<DIFFAUDIT_ROOT>/Download/white-box/`

Then removed the now-empty `<DIFFAUDIT_ROOT>/Research/external/downloads/` staging directory.

## Verification

Filesystem checks after the move:

- `<DIFFAUDIT_ROOT>/Research/external/downloads/` does not exist.
- `<DIFFAUDIT_ROOT>/Download/black-box/supplementary/recon-assets/` exists.
- `<DIFFAUDIT_ROOT>/Download/manifests/research-download-manifest.json` exists.
- `<DIFFAUDIT_ROOT>/Download/shared/datasets/` and `<DIFFAUDIT_ROOT>/Download/shared/weights/` exist under the root raw-intake tree.

Path search after the move:

- no active reference points to `Research/external/downloads`
- no active command example points to `external/CLiD/inter_output`
- remaining `external/recon-assets` mentions are historical audit/log context or explicit "no live command" notes

## Current Role Map

- `external/CLiD/` remains the code clone / `--repo-root`.
- `Download/black-box/supplementary/clid-mia-supplementary/` is the raw CLiD supplementary mirror.
- `external/SecMI/` remains the full upstream reference clone.
- `third_party/secmi/` remains the minimal in-repo integration surface.
- `Download/black-box/supplementary/recon-assets/` is the canonical raw recon asset bundle location.

## Handoff

No Platform or Runtime API change is required. System-consumable references should use the documented `Download` paths for raw assets and avoid reintroducing raw intake under `Research/external`.
