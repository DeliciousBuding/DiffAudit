# Paper Reading Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one rigorous reading report per indexed material, sync each report to an individual Feishu document, and link those documents back from the master DiffAudit index.

**Architecture:** Use a repo-side report specification as the single writing contract, generate one Markdown report per material under a normalized path, then create one Feishu document per report with public-read permissions. The master Feishu index remains the summary layer and links to each detailed report by title.

**Tech Stack:** Markdown, PowerShell, Python PDF tooling, `lark-cli`, Codex subagents

---

### Task 1: Define the Writing Contract

**Files:**
- Create: `docs/paper-reports/README.md`
- Create: `docs/paper-reports/report-spec.md`
- Create: `docs/paper-reports/report-template.md`

- [ ] Write the report specification
- [ ] Define section order, evidence requirements, and style constraints
- [ ] Define required metadata fields for each report
- [ ] Define the summary-extraction rules used to update `references/materials/paper-index.md`
- [ ] Commit

### Task 2: Create Output Layout and Tracking Manifest

**Files:**
- Create: `docs/paper-reports/manifest.csv`
- Create: `docs/paper-reports/black-box/.gitkeep`
- Create: `docs/paper-reports/gray-box/.gitkeep`
- Create: `docs/paper-reports/white-box/.gitkeep`
- Create: `docs/paper-reports/survey/.gitkeep`
- Create: `docs/paper-reports/context/.gitkeep`
- Create: `docs/paper-reports/assets/.gitkeep`

- [ ] Create normalized output directories
- [ ] Seed a manifest that maps each material to a future report file and Feishu doc slot
- [ ] Verify every indexed material is represented
- [ ] Commit

### Task 3: Dispatch Reading Agents

**Files:**
- Modify: `docs/paper-reports/manifest.csv`
- Create: `docs/paper-reports/<track>/<report>.md`

- [ ] Dispatch one subagent per indexed material
- [ ] Require each agent to follow the shared report spec exactly
- [ ] Require each agent to extract one reusable figure candidate and one method flow description
- [ ] Save one Markdown report per material
- [ ] Update manifest with completion state
- [ ] Commit in batches

### Task 4: Create Feishu Report Documents

**Files:**
- Modify: `docs/paper-reports/manifest.csv`

- [ ] Create one Feishu doc per report
- [ ] Apply public-read permissions to each doc
- [ ] Upload report Markdown into each doc
- [ ] Insert image assets when available
- [ ] Record document URL and permission state in the manifest
- [ ] Commit

### Task 5: Update Master Index

**Files:**
- Modify: `references/materials/paper-index.md`

- [ ] Replace current short summaries with report-derived scientific summaries
- [ ] Add title-style links to the single-paper Feishu documents
- [ ] Verify the master Feishu doc reflects the updated repo-side summaries
- [ ] Commit

### Task 6: Verify End-to-End Consistency

**Files:**
- Modify: `docs/paper-reports/manifest.csv`

- [ ] Check every indexed material has a local report file
- [ ] Check every report has a Feishu document URL
- [ ] Check every Feishu document is public-readable by link
- [ ] Check the master Feishu doc links back to every single-paper report
- [ ] Run a final repo status check
- [ ] Commit
