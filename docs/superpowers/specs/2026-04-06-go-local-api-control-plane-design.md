# Go Local API Control Plane Design

## Goal

Replace the current Python FastAPI local service with a Go-based control plane that keeps the same HTTP contract while continuing to delegate real experiment execution to the existing Python research CLI.

This design is intentionally narrow:

- move only the local API control plane to Go
- do not rewrite `recon` or any research execution logic in Go
- keep JSON-based summaries and job records as the source of truth

## Why This Scope

The current local API process is a lightweight HTTP shell around:

- reading `experiments/*/summary.json`
- writing `workspaces/local-api/jobs/*.json`
- spawning `python -m diffaudit ...`

The memory-heavy part is the Python execution chain behind each job, not the HTTP surface itself. Rewriting the control plane in Go reduces steady-state memory and startup overhead without disrupting the existing research workflow.

## Chosen Approach

Build a standalone Go service under `..\Services\Local-API/` and keep the Python CLI as the worker backend.

This is the shortest correct path because it:

- avoids adding another Python resident process
- preserves all existing experiment artifacts and summaries
- avoids porting `torch/diffusers` workflows
- allows the platform to keep the same local API contract

## Rejected Alternatives

### Keep FastAPI and only micro-optimize Python

Rejected because it does not materially improve the main concern: reducing the resident service footprint.

### Rewrite both API and execution chain in Go

Rejected because it explodes scope and would require reimplementing research logic that already works.

### Use Rust instead of Go

Rejected for now because Rust is not installed on this machine, while Go is already available and is sufficient for this control-plane workload.

## Architecture

The Go service becomes the single local API server. Python remains the execution backend.

### Go service responsibilities

- expose the existing local HTTP endpoints
- read summary files from `experiments/`
- read and write job records under `workspaces/local-api/jobs/`
- enforce workspace-level active job exclusion
- spawn and monitor `python -m diffaudit ...` child processes

### Python responsibilities

- continue to implement `run-recon-artifact-mainline`
- continue to implement `run-recon-runtime-mainline`
- continue to produce `summary.json` and artifact outputs

## Directory Layout

New code lives in:

- `..\Services\Local-API/go.mod`
- `..\Services\Local-API/cmd/local-api/main.go`
- `..\Services\Local-API/internal/...`

The existing Python local API module remains temporarily for compatibility during migration, but it should stop being the preferred control-plane entry once the Go service is verified.

## HTTP Contract

The Go service must preserve these endpoints:

- `GET /health`
- `GET /api/v1/models`
- `GET /api/v1/experiments/recon/best`
- `GET /api/v1/experiments/{workspace}/summary`
- `GET /api/v1/audit/jobs`
- `POST /api/v1/audit/jobs`
- `GET /api/v1/audit/jobs/{job_id}`

The response shape should remain stable enough that the platform does not need a parallel contract update.

## Data Sources

### Experiment summaries

Read directly from:

- `experiments/*/summary.json`
- `experiments/blackbox-status/summary.json`

`recon/best` continues to prefer the aggregate black-box status file and falls back to scanning experiment summaries only if the aggregate file is missing.

### Job records

Continue to persist job state to:

- `workspaces/local-api/jobs/<job_id>.json`

The Go service should remain compatible with the current JSON shape so existing tools and debugging habits still work.

## Job Execution Flow

1. Validate request payload.
2. Reject if another `queued` or `running` job already exists for the same `workspace_name`.
3. Write the initial job record with `queued` status.
4. Spawn a goroutine to execute the Python CLI command.
5. Update the job record to `running`.
6. Wait for the child process to finish.
7. If `<workspace>/summary.json` exists and exit code is zero, mark the job `completed` and extract headline metrics.
8. Otherwise mark the job `failed` and preserve stdout/stderr tails.

## Concurrency Model

The Go service does not need a complex queue.

Use:

- one goroutine per submitted job
- file-based state as the persisted source of truth
- a process-local mutex around job creation and job-file updates to avoid local races

This is sufficient because the service is local-machine scoped and current workload is low concurrency.

## Error Handling

### Request validation errors

Return `400` for malformed input.

### Workspace conflicts

Return `409` when the same `workspace_name` already has an active job.

### Missing summaries or jobs

Return `404`.

### Backend execution failure

Persist the failure to the job record and return it through `GET /api/v1/audit/jobs/{job_id}`.

## Performance Expectations

Expected improvements:

- lower resident memory for the always-on API process
- faster startup than the current Python web server
- less framework overhead for simple JSON/file/process operations

Non-goals:

- reducing `torch/diffusers` runtime memory during experiment execution
- speeding up the actual model inference path

## Testing Plan

Implement test-first in Go.

Minimum coverage:

- health endpoint
- model listing
- best `recon` summary selection
- workspace summary lookup
- job creation
- job listing order
- duplicate workspace rejection
- job detail lookup

Worker execution can be tested with stub commands and temporary directories before wiring in the real Python commands.

## Rollout Plan

1. Add the Go service with tests.
2. Make it runnable with a single command from the repo root.
3. Verify it against the existing local API contract.
4. Update local API documentation to make the Go service the primary entrypoint.
5. Keep the Python local API code only as a temporary fallback until the Go control plane is stable.

## Acceptance Criteria

- Go service starts successfully on this machine
- existing local API contract is preserved
- job records remain JSON-compatible
- `recon_artifact_mainline` and `recon_runtime_mainline` still execute through Python CLI
- platform can talk to the Go service without a second contract redesign

