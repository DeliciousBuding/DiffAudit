# Licensing

DiffAudit Research uses a permissive open-source license for first-party
project work while keeping third-party materials under their original terms.

This is a project policy note, not legal advice.

## Project License

Unless a file or directory says otherwise, first-party DiffAudit Research source
code, configuration templates, tests, scripts, and original project
documentation are licensed under the [Apache License 2.0](../LICENSE).

Why Apache 2.0:

- It is permissive, so teammates and downstream services can use, modify, and
  redistribute the code with low friction.
- It includes an explicit patent grant and patent-termination clause, which is
  safer for research code that may later connect to product or runtime systems.
- It is clearer for outside reviewers than leaving the repository unlicensed or
  relying on an informal README note.

## What Is Not Relicensed

The project license does not relicense third-party or external materials. In
particular, it does not cover:

- datasets, model weights, checkpoints, or supplementary bundles under
  `<DIFFAUDIT_ROOT>/Download/`;
- ignored upstream code clones under `external/`;
- paper PDFs, extracted figures, OCR output, or publisher-provided materials in
  `references/` or `docs/paper-reports/`;
- model cards, gated Hugging Face assets, or assets with upstream terms;
- vendored third-party code that carries its own license notice.

Always follow the upstream license, terms of use, and access rules for those
materials.

## Vendored Code

| Location | Source | Upstream license | Local note |
| --- | --- | --- | --- |
| `third_party/secmi/` | <https://github.com/jinhaoduan/SecMI> | MIT | Minimal adapter-facing subset; retained at [third_party/secmi/LICENSE](../third_party/secmi/LICENSE) |

When adding vendored code, keep the upstream source URL, copyright notice, and
license text with the vendored subset.

## Contribution Licensing

Unless you explicitly mark a submission as "Not a Contribution", contributions
intentionally submitted to this repository are accepted under Apache 2.0, as
described in section 5 of the license.

Do not contribute third-party code, datasets, weights, paper assets, or
generated artifacts unless the source and redistribution terms are clear enough
to record in this repository.
