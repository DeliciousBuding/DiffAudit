# Third-Party Vendored Code

`third_party/` 只放当前仓库真实依赖、且需要一起维护的最小 vendored 子集。

## Current Role Map

- `secmi/`
  - canonical in-repo minimal integration surface for current SecMI adapter and commands
  - if you need the full upstream repository layout, config files, or upstream reference tree, use `D:\Code\DiffAudit\Research\external\SecMI\`

## Not For

- full upstream clones
- raw downloads
- run outputs
- large checkpoints or datasets
