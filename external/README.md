# External Code Clones

`external/` 只放上游代码仓或本地 exploratory clone。

不要把原始数据集、原始权重、补充附件长期堆在这里。

## Current Role Map

- `CLiD/`
  - canonical local working clone for upstream CLiD code
  - raw CLiD supplementary mirror lives under `D:\Code\DiffAudit\Download\black-box\supplementary\clid-mia-supplementary\`
- `SecMI/`
  - full upstream clone kept for upstream config / split / structure reference
  - canonical in-repo minimal integration surface is `D:\Code\DiffAudit\Research\third_party\secmi\`
- `PIA/`, `Reconstruction-based-Attack/`, `DiT/`, `DPDM/`, `mia-diffusion/`
  - upstream or exploratory code clones

## Not For

- raw downloaded asset bundles
- lane-normalized admitted assets
- run evidence

Use:

- `D:\Code\DiffAudit\Download\` for raw intake
- `D:\Code\DiffAudit\Research\workspaces\<lane>\assets\` for lane-normalized gateways
- `D:\Code\DiffAudit\Research\workspaces\<lane>\runs\` for evidence
