# Research Storage Layout Finalization

Date: 2026-04-26

Verdict: `positive`

## Decision

The storage boundary cleanup is now physically aligned with the documented rule:

- `D:\Code\DiffAudit\Research\external\` is for upstream or exploratory code clones.
- `D:\Code\DiffAudit\Research\third_party\` is for minimal vendored integration code.
- `D:\Code\DiffAudit\Download\` is the raw intake layer for datasets, weights, supplementary bundles, manifests, and author release mirrors.
- `D:\Code\DiffAudit\Research\workspaces\<lane>\assets\` is for lane-normalized gateways and manifests.
- `D:\Code\DiffAudit\Research\workspaces\<lane>\runs\` is for evidence and run outputs.

## Physical Cleanup

Moved the raw intake tree out of the Research external-code surface:

- from `D:\Code\DiffAudit\Research\external\downloads\black-box\`
- from `D:\Code\DiffAudit\Research\external\downloads\gray-box\`
- from `D:\Code\DiffAudit\Research\external\downloads\manifests\`
- from `D:\Code\DiffAudit\Research\external\downloads\shared\`
- from `D:\Code\DiffAudit\Research\external\downloads\white-box\`

to:

- `D:\Code\DiffAudit\Download\black-box\`
- `D:\Code\DiffAudit\Download\gray-box\`
- `D:\Code\DiffAudit\Download\manifests\`
- `D:\Code\DiffAudit\Download\shared\`
- `D:\Code\DiffAudit\Download\white-box\`

Then removed the now-empty `D:\Code\DiffAudit\Research\external\downloads\` staging directory.

## Verification

Filesystem checks after the move:

- `D:\Code\DiffAudit\Research\external\downloads\` does not exist.
- `D:\Code\DiffAudit\Download\black-box\supplementary\recon-assets\` exists.
- `D:\Code\DiffAudit\Download\manifests\research-download-manifest.json` exists.
- `D:\Code\DiffAudit\Download\shared\datasets\` and `D:\Code\DiffAudit\Download\shared\weights\` exist under the root raw-intake tree.

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
