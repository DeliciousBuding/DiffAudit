# 2026-04-16 Black-Box Prompt-Response Consistency Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-1.4 / prompt-response consistency probe`
- `method_family`: `prompt-response-consistency`
- `device`: `cpu`
- `decision`: `negative but useful`

## Question

On the current local `semantic-auxiliary-classifier` target-family outputs, does prompt-to-returned-image consistency expose a usable black-box membership signal, or is it too weak to matter relative to the existing returned-image similarity line?

## Executed Evidence

Primary run artifact:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\prompt-response-consistency-20260416-r1\summary.json`

Input artifacts reused:

- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260416-r2\outputs\records.json`
- `D:\Code\DiffAudit\Research\workspaces\black-box\runs\semantic-aux-classifier-comparator-20260416-r2\outputs\generated`

## Metrics

Observed prompt-response consistency metrics on `32 / 32`:

- `AUC = 0.481445`
- `ASR = 0.578125`
- `TPR@1%FPR = 0.0`

Relative to current semantic-aux reference:

- `Spearman(prompt_consistency, mean_cos) = 0.109112`
- `AUC gain vs mean_cos = -0.435547`

Class means:

- member mean prompt-cos = `0.196779`
- non-member mean prompt-cos = `0.199764`

## Verdict

Current verdict:

- `negative but useful`

Reason:

1. the prompt-response score is effectively near-random under the current local protocol;
2. it does not merely fail to beat `mean_cos`; it underperforms it heavily;
3. rank agreement with `mean_cos` is also very low, so this is not a hidden refinement of the current challenger either;
4. the useful part is that this closes one plausible black-box hypothesis without spending a new generation run.

## Decision

Current decision:

- close `BB-1.4` as `negative but useful`
- keep `semantic-auxiliary-classifier` centered on returned-image similarity, not prompt-response consistency
- do not reopen prompt-response consistency unless a different generation protocol or stronger text-conditioning setup is introduced

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: if mentioned, the honest wording is that prompt-response consistency was checked and did not yield a usable black-box signal on the current local stack.
