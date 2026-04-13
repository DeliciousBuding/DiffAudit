# Paper Reading Reports

This directory stores one detailed reading report per indexed material under `references/materials/`.

## Goals

- Use one uniform scientific writing standard across all reports.
- Preserve one local Markdown report per material in the repository.
- Maintain one Feishu document per report for link-based reading and review.
- Feed the concise, scientific summary in `references/materials/paper-index.md` from the corresponding detailed report rather than hand-written ad hoc notes.

## Layout

```text
docs/paper-reports/
  README.md
  report-spec.md
  report-template.md
  manifest.csv
  black-box/
  gray-box/
  white-box/
  survey/
  context/
  assets/
```

## Naming Rules

- Report file name: reuse the PDF file stem and append `-report.md`.
- Report location: place reports under the track folder that matches `references/materials/<track>/`.
- Image assets: store extracted figures under `docs/paper-reports/assets/<track>/`.
- Key figure image name: `<pdf-stem>-key-figure-p<page>.png`.
- If the figure is cropped from a page region, keep the same naming rule and record the crop box inside the report.

## Feishu Rules

This repo only defines the report content contract.

Workspace-local Feishu publishing rules and commands now live outside `Research`:

- `D:\Code\DiffAudit\LocalOps\feishu\README.md`
- `D:\Code\DiffAudit\LocalOps\feishu\docs\workspace-rules.md`
- `D:\Code\DiffAudit\LocalOps\feishu\docs\source-map.md`

Within `Research`, keep these expectations:

- One report corresponds to one Feishu doc.
- Each Feishu doc must be readable by anyone with the link.
- The master Feishu doc should link to the single-report Feishu docs by title.
