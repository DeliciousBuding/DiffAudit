# 2026-04-16 Black-Box Next-Lane Score-Package Selection Verdict

## Task

- select the next fresh black-box execution-ready lane after current branch closures

## Question

- After `Recon` remains the headline and `semantic-auxiliary-classifier` remains the leading challenger, what is the shortest honest black-box reopen path that advances the model line without inventing a fake new family or wasting GPU?

## Evidence Base

- `workspaces/black-box/2026-04-15-blackbox-second-signal-semantic-aux-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-semantic-aux-scoring-verdict.md`
- `workspaces/black-box/2026-04-16-blackbox-semantic-aux-fusion-verdict.md`
- `workspaces/black-box/2026-04-10-recon-decision-package.md`

## Verdict

- `positive selection`

The next lane should be:

- `BB-6 same-protocol cross-method score package`

## Why This Lane Wins

1. current black-box truth already allows this exact reopen shape:
   - `BB-2` may reopen with `a new feature family or a same-protocol cross-method score package`
2. it is shorter and more honest than claiming a brand-new family before proving one;
3. it directly leverages the two strongest current black-box assets:
   - `Recon` as headline
   - `semantic-auxiliary-classifier` as leading challenger
4. it can start entirely on CPU-side existing artifacts, so it does not need a fresh GPU admission question yet.

## Decision

Current decision:

- open `BB-6`
- keep first pass CPU-only
- defer any GPU request until the score-package shows bounded gain over the best single method on aligned artifacts

## Handoff Note

- `Platform`: no sync needed.
- `Runtime`: no sync needed.
- `Materials`: no immediate sync needed; if mentioned, describe this only as the next black-box candidate package, not a new admitted result.
