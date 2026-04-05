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

## Feishu Rules

- One report corresponds to one Feishu doc.
- Each Feishu doc must be readable by anyone with the link.
- The master Feishu doc should link to the single-report Feishu docs by title.
- If a figure asset is available, insert it into the single-report Feishu doc after the main summary section.
