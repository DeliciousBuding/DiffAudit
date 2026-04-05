# Paper Reading Report Specification

## Purpose

This specification defines the required structure, style, and evidence standard for all paper reading reports generated for DiffAudit.

The reports are intended for internal technical review, future onboarding, and direct reuse in Feishu-facing project updates. They must therefore read like scientific technical notes rather than casual summaries.

## Writing Style

- Use scientific, explicit, non-metaphorical language.
- Avoid colloquialisms, hype, vague praise, and conversational filler.
- Distinguish clearly between claims made by the paper and inferences made by the report author.
- When a paper leaves ambiguity, say so explicitly rather than smoothing it over.
- Use complete sentences and precise terminology.
- Prefer evidence-backed wording such as `the paper reports`, `the authors assume`, `the evaluation shows`, `the current report infers`.
- When the paper relies on equations, quote the essential equations using LaTeX math blocks or inline LaTeX rather than replacing them with vague prose.

## Required Metadata Block

Every report must begin with the following metadata bullets:

- `Title`
- `Material Path`
- `Primary Track`
- `Venue / Year`
- `Threat Model Category`
- `Core Task`
- `Open-Source Implementation`
- `Report Status`

## Required Sections

Every report must contain all sections below in this exact order.

### 1. Executive Summary

- 2 to 4 paragraphs
- State the problem, the main method, the main result, and why the paper matters to DiffAudit

### 2. Bibliographic Record

- Title
- Authors if available
- Venue / year / version
- Local PDF path
- Source URL if known

### 3. Research Question

- What exact question is the paper trying to answer
- What threat model or deployment setting it assumes

### 4. Problem Setting and Assumptions

- Access model
- Available inputs
- Available outputs
- Required priors or side information
- Scope limits

### 5. Method Overview

- Plain-language walkthrough of the method
- What signal the method exploits
- What must happen step by step

### 6. Method Flow

- One Mermaid flowchart in fenced code block
- Show the method pipeline from input assumptions to output decision

### 7. Key Technical Details

- Important equations, losses, features, or statistics in words
- Any notable optimization or scoring detail
- Any distinctive implementation requirement
- Include 1 to 3 core formulas in LaTeX form when the paper has explicit mathematical definitions worth preserving

### 8. Experimental Setup

- Datasets
- Model families
- Baselines
- Metrics
- Evaluation conditions

### 9. Main Results

- What the paper reports as its main findings
- Which claims appear strongest
- Which results depend heavily on the setting

### 10. Strengths

- Concrete technical or experimental strengths only

### 11. Limitations and Validity Threats

- Missing ablations
- Unclear assumptions
- Reproducibility concerns
- Possible over-claiming

### 12. Reproducibility Assessment

- What assets are needed
- Whether code exists
- Whether the current DiffAudit repository already covers part of the route
- What blocks faithful reproduction today

### 13. Relevance to DiffAudit

- How this paper maps to current routes, assets, product direction, or report narrative

### 14. Recommended Figure

- Identify one figure, table, or page worth surfacing
- Explain why it is worth surfacing
- Record page number when possible
- Render the chosen figure or table region into a PNG asset
- Inspect the rendered image directly before finalizing the explanation
- Include the local image path in this section
- Prefer a cropped figure region over a full-page screenshot if the page contains substantial surrounding body text
- If a full-page fallback is unavoidable, say why a clean region crop was not feasible

### 15. Extracted Summary for `paper-index.md`

- Write exactly three paragraphs:
- Paragraph 1: what problem the paper addresses
- Paragraph 2: what method or core conclusion it contributes
- Paragraph 3: why it matters to DiffAudit specifically

This section is the only section allowed to be reused in shortened form inside `references/materials/paper-index.md`.

## Minimum Depth Standard

- Target 1200 to 2200 Chinese characters per report body, excluding metadata.
- Reports for foundational or mainline papers may be longer.
- The report must be based on actual reading of the PDF, not title inference alone.

## Figure and Diagram Standard

- If a representative figure can be identified, record the page number in `Recommended Figure`.
- The report author must render at least one key figure or table region into a PNG asset and inspect that image directly instead of relying on text extraction alone.
- Use the normalized asset path pattern `docs/paper-reports/assets/<track>/<pdf-stem>-key-figure-p<page>.png`.
- Insert the image into the local Markdown report immediately after the `Recommended Figure` section using a relative Markdown image link.
- Prefer cropped regions that isolate the actual chart, figure, or table rather than full-page screenshots with large text blocks.
- Every report must include a Mermaid flowchart even if no image is extracted.
- Figures should be descriptive supplements, not substitutes for analysis.

## Prohibited Shortcuts

- Do not summarize from title alone.
- Do not use generic phrases such as `很有启发`, `值得关注`, `效果很好` unless followed by precise evidence.
- Do not collapse limitations into praise.
- Do not copy the abstract as the report.
