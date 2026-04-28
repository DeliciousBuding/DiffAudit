# Security Policy

DiffAudit Research is a research repository for privacy-risk auditing of
diffusion models. It contains experimental code, research notes, and
reproducibility scaffolding. It is not the production Runtime service or the
public Platform application.

## Supported Scope

Please report security issues that affect this repository's first-party code,
configuration templates, tests, scripts, GitHub workflows, or documentation
that could lead to unsafe execution, credential leakage, or misleading security
claims.

Examples:

- a workflow or script that may expose secrets;
- a path-handling bug that can overwrite files outside the workspace;
- unsafe loading of untrusted artifacts in first-party code;
- a documented command that encourages publishing private data, model weights,
  or credentials;
- a reproducibility claim that could materially mislead downstream Runtime or
  Platform consumers.

## Out Of Scope

The following are not handled as vulnerabilities in this repository:

- vulnerabilities in upstream datasets, model weights, papers, or ignored
  `external/` clones;
- missing access to gated third-party assets;
- model-performance disagreements or unvalidated research hypotheses;
- reports about the sibling `Runtime-Server/` or `Platform/` repositories,
  unless the issue is caused by a Research artifact or handoff contract.

## Reporting

For private security reports, use GitHub private vulnerability reporting if it
is available for the repository. If that is not available, open a minimal public
issue that says a private security report is needed, without including secrets,
exploit details, private data, or unpublished asset links.

Do not post credentials, private dataset paths, proprietary model weights, or
attack payloads in public issues, pull requests, screenshots, or logs.

## Response Expectations

DiffAudit Research is maintained as a research prototype. Response times may
vary, but reports with clear affected files, commands, reproduction steps, and
impact will be prioritized. When a report affects Runtime or Platform
consumers, the fix should include a handoff note in the relevant documentation.
