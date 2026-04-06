# Go Local API Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python-resident local API server with a Go control plane that preserves the current HTTP contract while continuing to run research jobs through the Python CLI.

**Architecture:** Add a standalone Go service under `tools/local-api-go/` using the standard `net/http` stack. Keep `experiments/*/summary.json` and `workspaces/local-api/jobs/*.json` as the persisted source of truth, and continue to spawn `python -m diffaudit ...` for `recon` job execution.

**Tech Stack:** Go 1.26, standard library `net/http`, `encoding/json`, `os/exec`, Python CLI backend, existing JSON artifact schema

---

### Task 1: Scaffold The Go Service And Preserve The HTTP Contract

**Files:**
- Create: `D:\Code\DiffAudit\Project\tools\local-api-go\go.mod`
- Create: `D:\Code\DiffAudit\Project\tools\local-api-go\cmd\local-api\main.go`
- Create: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server.go`
- Create: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server_test.go`

- [ ] **Step 1: Write the failing tests for `GET /health`, `GET /api/v1/models`, `GET /api/v1/experiments/recon/best`, and `GET /api/v1/experiments/{workspace}/summary`**

The tests should use `httptest`, temporary directories, and synthetic `summary.json` payloads that match the current Python API contract.

- [ ] **Step 2: Run the Go tests to verify they fail because the service does not exist yet**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: compile failure or missing symbol failure for the new server package.

- [ ] **Step 3: Write the minimal Go server code to make the read-only endpoints pass**

Implement:

- config for `experiments_root` and `jobs_root`
- static model catalog
- best `recon` summary selection
- workspace summary loading

- [ ] **Step 4: Re-run the Go tests and verify the read-only endpoints pass**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: PASS for the endpoint tests created in this task.

- [ ] **Step 5: Commit**

```powershell
git -C D:\Code\DiffAudit\Project add tools/local-api-go
git -C D:\Code\DiffAudit\Project commit -m "Scaffold Go local API service"
```

### Task 2: Port Job List/Create/Get Behavior To Go

**Files:**
- Modify: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server.go`
- Modify: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server_test.go`

- [ ] **Step 1: Write the failing tests for `GET /api/v1/audit/jobs`, `POST /api/v1/audit/jobs`, and `GET /api/v1/audit/jobs/{job_id}`**

Include tests for:

- persisted job records
- latest-first job listing
- duplicate active `workspace_name` rejection with `409`

- [ ] **Step 2: Run the Go tests to verify the new job tests fail**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: FAIL for missing or incorrect job route behavior.

- [ ] **Step 3: Implement JSON job persistence and workspace conflict checks**

Implement:

- job record file writes
- job list reads
- descending sort by creation time
- duplicate active workspace rejection

- [ ] **Step 4: Re-run the Go tests and verify the job behavior passes**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: PASS for job list/create/get tests.

- [ ] **Step 5: Commit**

```powershell
git -C D:\Code\DiffAudit\Project add tools/local-api-go
git -C D:\Code\DiffAudit\Project commit -m "Add Go local API job persistence"
```

### Task 3: Wire Go Job Execution To The Python CLI

**Files:**
- Modify: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server.go`
- Modify: `D:\Code\DiffAudit\Project\tools\local-api-go\internal\api\server_test.go`

- [ ] **Step 1: Write the failing tests for background job execution and completion state updates**

Use a stub executable or a controlled `python` command override to verify:

- job transitions `queued -> running -> completed`
- `summary_path` and `metrics` are filled after completion
- failed commands persist `stderr_tail` and `error`

- [ ] **Step 2: Run the Go tests to verify the execution tests fail**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: FAIL because background execution and record updates are not implemented yet.

- [ ] **Step 3: Implement background command execution through `python -m diffaudit ...`**

Keep the command builders compatible with:

- `recon_artifact_mainline`
- `recon_runtime_mainline`

- [ ] **Step 4: Re-run the Go tests and verify execution behavior passes**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: PASS for background execution tests.

- [ ] **Step 5: Commit**

```powershell
git -C D:\Code\DiffAudit\Project add tools/local-api-go
git -C D:\Code\DiffAudit\Project commit -m "Wire Go local API jobs to Python CLI"
```

### Task 4: Make The Go Service The Primary Local Entry Point

**Files:**
- Modify: `D:\Code\DiffAudit\Project\README.md`
- Modify: `D:\Code\DiffAudit\Project\docs\local-api.md`
- Modify: `D:\Code\DiffAudit\Project\docs\reproduction-status.md`

- [ ] **Step 1: Write the failing doc expectation test or checklist**

Use a lightweight verification checklist instead of a code test:

- README points to the Go service as the primary startup path
- `docs/local-api.md` documents the Go binary/service
- status docs state that the Go control plane is now the preferred entrypoint

- [ ] **Step 2: Update the docs with the Go-first startup path**

Document:

- how to run `go run ./cmd/local-api`
- which directories it reads
- that Python remains the execution backend

- [ ] **Step 3: Verify the docs and service startup command manually**

Run:

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go run ./cmd/local-api --help
```

Expected: command usage output renders successfully.

- [ ] **Step 4: Commit**

```powershell
git -C D:\Code\DiffAudit\Project add README.md docs/local-api.md docs/reproduction-status.md
git -C D:\Code\DiffAudit\Project commit -m "Document Go local API control plane"
```

### Task 5: End-To-End Verification

**Files:**
- Modify if needed: `D:\Code\DiffAudit\Project\docs\local-api.md`

- [ ] **Step 1: Run the full Go test suite**

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go test ./...
```

Expected: PASS

- [ ] **Step 2: Run the existing Python local API regression tests that remain relevant to contract shape**

```powershell
cd D:\Code\DiffAudit\Project
conda run -n diffaudit-research python -m unittest tests.test_local_api
```

Expected: PASS or intentionally retired tests updated to reflect the Go primary entrypoint while preserving the HTTP contract.

- [ ] **Step 3: Run a manual local smoke**

```powershell
cd D:\Code\DiffAudit\Project\tools\local-api-go
go run ./cmd/local-api
```

Then verify:

```powershell
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/api/v1/models
```

- [ ] **Step 4: Commit final cleanup if needed**

```powershell
git -C D:\Code\DiffAudit\Project add .
git -C D:\Code\DiffAudit\Project commit -m "Finish Go local API control plane migration"
```
