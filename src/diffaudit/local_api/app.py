"""FastAPI app that exposes the local DiffAudit research workflows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from diffaudit.reports.blackbox_status import (
    BLACKBOX_EVIDENCE_RANK,
    _evidence_sample_count,
    _extract_headline_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "workspaces" / "local-api" / "jobs"
DEFAULT_RECON_REPO_ROOT = PROJECT_ROOT / "external" / "Reconstruction-based-Attack"

JobStatus = Literal["queued", "running", "completed", "failed"]
JobType = Literal["recon_artifact_mainline", "recon_runtime_mainline"]

MODEL_OPTIONS = [
    {
        "key": "sd15-ddim",
        "label": "Stable Diffusion 1.5 + DDIM",
        "access_level": "black-box",
        "availability": "ready",
        "paper": "BlackBox_Reconstruction_ArXiv2023",
        "method": "recon",
        "backend": "stable_diffusion",
        "scheduler": "ddim",
    },
    {
        "key": "kandinsky-v22",
        "label": "Kandinsky v2.2",
        "access_level": "black-box",
        "availability": "partial",
        "paper": "BlackBox_Reconstruction_ArXiv2023",
        "method": "recon",
        "backend": "kandinsky_v22",
        "scheduler": None,
    },
    {
        "key": "dit-xl2-256",
        "label": "DiT-XL/2 256",
        "access_level": "black-box",
        "availability": "partial",
        "paper": "Scalable_Diffusion_Transformers_2022",
        "method": "sample",
        "backend": "dit",
        "scheduler": None,
    },
]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    experiments_root: str
    jobs_root: str


class ModelOption(BaseModel):
    key: str
    label: str
    access_level: Literal["black-box", "gray-box", "white-box"]
    availability: Literal["ready", "partial", "planned"]
    paper: str
    method: str
    backend: str
    scheduler: str | None = None


class ExperimentSummaryEnvelope(BaseModel):
    status: str
    paper: str | None = None
    method: str | None = None
    mode: str | None = None
    backend: str | None = None
    scheduler: str | None = None
    workspace: str
    metrics: dict[str, object]
    artifact_paths: dict[str, object]
    summary_path: str
    raw: dict[str, object]


class AuditJobCreate(BaseModel):
    job_type: JobType
    workspace_name: str = Field(..., min_length=1)
    repo_root: str | None = None
    method: str = "threshold"
    artifact_dir: str | None = None
    target_member_dataset: str | None = None
    target_nonmember_dataset: str | None = None
    shadow_member_dataset: str | None = None
    shadow_nonmember_dataset: str | None = None
    target_model_dir: str | None = None
    shadow_model_dir: str | None = None
    target_decoder_dir: str | None = None
    target_prior_dir: str | None = None
    shadow_decoder_dir: str | None = None
    shadow_prior_dir: str | None = None
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    num_validation_images: int = 3
    inference_steps: int = 30
    gpu: int = 0
    backend: str = "stable_diffusion"
    scheduler: str = "default"
    similarity_method: str = "cosine"
    image_encoder: str = "deit"

    @model_validator(mode="after")
    def validate_payload(self) -> "AuditJobCreate":
        if "/" in self.workspace_name or "\\" in self.workspace_name:
            raise ValueError("workspace_name must be a single workspace directory name")

        if self.job_type == "recon_artifact_mainline":
            if not self.artifact_dir:
                raise ValueError("recon_artifact_mainline requires artifact_dir")
            return self

        required_fields = (
            "target_member_dataset",
            "target_nonmember_dataset",
            "shadow_member_dataset",
            "shadow_nonmember_dataset",
            "target_model_dir",
            "shadow_model_dir",
        )
        missing = [field for field in required_fields if not getattr(self, field)]
        if missing:
            raise ValueError(
                "recon_runtime_mainline missing required fields: " + ", ".join(missing)
            )
        return self


class AuditJobResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    workspace_name: str
    created_at: str
    updated_at: str
    payload: dict[str, object]
    command: list[str] | None = None
    summary_path: str | None = None
    metrics: dict[str, object] | None = None
    error: str | None = None
    stdout_tail: list[str] | None = None
    stderr_tail: list[str] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_file(jobs_root: Path, job_id: str) -> Path:
    return jobs_root / f"{job_id}.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        raise FileNotFoundError(f"summary not found: {summary_path}")
    return _read_json(summary_path)


def _summary_envelope(summary_path: Path) -> ExperimentSummaryEnvelope:
    payload = _load_summary(summary_path)
    runtime = payload.get("runtime", {})
    backend = None
    scheduler = None
    if isinstance(runtime, dict):
        backend = runtime.get("backend")
        scheduler = runtime.get("scheduler")
    return ExperimentSummaryEnvelope(
        status=str(payload.get("status", "unknown")),
        paper=str(payload.get("paper")) if payload.get("paper") else None,
        method=str(payload.get("method")) if payload.get("method") else None,
        mode=str(payload.get("mode")) if payload.get("mode") else None,
        backend=str(backend) if backend else None,
        scheduler=str(scheduler) if scheduler else None,
        workspace=str(payload.get("workspace", str(summary_path.parent))),
        metrics=dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {},
        artifact_paths=(
            dict(payload.get("artifact_paths", {}))
            if isinstance(payload.get("artifact_paths"), dict)
            else {}
        ),
        summary_path=str(summary_path),
        raw=payload,
    )


def _iter_blackbox_summaries(experiments_root: Path) -> list[tuple[Path, dict[str, object]]]:
    results: list[tuple[Path, dict[str, object]]] = []
    for summary_path in sorted(experiments_root.glob("*/summary.json")):
        payload = _read_json(summary_path)
        if payload.get("track") == "black-box" and payload.get("method"):
            results.append((summary_path, payload))
    return results


def _best_recon_summary_path(experiments_root: Path) -> Path:
    status_summary_path = experiments_root / "blackbox-status" / "summary.json"
    if status_summary_path.exists():
        status_payload = _read_json(status_summary_path)
        methods = status_payload.get("methods", {})
        if isinstance(methods, dict):
            recon_entry = methods.get("recon")
            if isinstance(recon_entry, dict):
                best_path = recon_entry.get("best_evidence_path")
                if isinstance(best_path, str) and Path(best_path).exists():
                    return Path(best_path)

    candidates: list[tuple[int, int, Path]] = []
    for summary_path, payload in _iter_blackbox_summaries(experiments_root):
        if payload.get("method") != "recon":
            continue
        mode = str(payload.get("mode", ""))
        rank = BLACKBOX_EVIDENCE_RANK.get(mode, 0)
        sample_count = _evidence_sample_count(payload)
        candidates.append((rank, sample_count, summary_path))

    if not candidates:
        raise FileNotFoundError("no recon experiment summaries found")

    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[-1][2]


def _create_job_record(jobs_root: Path, payload: AuditJobCreate) -> dict[str, object]:
    timestamp = _now_iso()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    record = {
        "job_id": job_id,
        "job_type": payload.job_type,
        "status": "queued",
        "workspace_name": payload.workspace_name,
        "created_at": timestamp,
        "updated_at": timestamp,
        "payload": payload.model_dump(),
        "command": None,
        "summary_path": None,
        "metrics": None,
        "error": None,
        "stdout_tail": None,
        "stderr_tail": None,
    }
    _write_json(_job_file(jobs_root, job_id), record)
    return record


def _update_job(jobs_root: Path, job_id: str, **changes: object) -> dict[str, object]:
    job_path = _job_file(jobs_root, job_id)
    record = _read_json(job_path)
    record.update(changes)
    record["updated_at"] = _now_iso()
    _write_json(job_path, record)
    return record


def _job_command(payload: AuditJobCreate, workspace_path: Path) -> list[str]:
    repo_root = payload.repo_root or str(DEFAULT_RECON_REPO_ROOT)
    if payload.job_type == "recon_artifact_mainline":
        return [
            sys.executable,
            "-m",
            "diffaudit",
            "run-recon-artifact-mainline",
            "--artifact-dir",
            str(payload.artifact_dir),
            "--workspace",
            str(workspace_path),
            "--repo-root",
            str(repo_root),
            "--method",
            payload.method,
        ]

    command = [
        sys.executable,
        "-m",
        "diffaudit",
        "run-recon-runtime-mainline",
        "--target-member-dataset",
        str(payload.target_member_dataset),
        "--target-nonmember-dataset",
        str(payload.target_nonmember_dataset),
        "--shadow-member-dataset",
        str(payload.shadow_member_dataset),
        "--shadow-nonmember-dataset",
        str(payload.shadow_nonmember_dataset),
        "--target-model-dir",
        str(payload.target_model_dir),
        "--shadow-model-dir",
        str(payload.shadow_model_dir),
        "--workspace",
        str(workspace_path),
        "--repo-root",
        str(repo_root),
        "--pretrained-model-name-or-path",
        payload.pretrained_model_name_or_path,
        "--num-validation-images",
        str(payload.num_validation_images),
        "--inference-steps",
        str(payload.inference_steps),
        "--gpu",
        str(payload.gpu),
        "--backend",
        payload.backend,
        "--scheduler",
        payload.scheduler,
        "--method",
        payload.method,
        "--similarity-method",
        payload.similarity_method,
        "--image-encoder",
        payload.image_encoder,
    ]
    optional_pairs = (
        ("--target-decoder-dir", payload.target_decoder_dir),
        ("--target-prior-dir", payload.target_prior_dir),
        ("--shadow-decoder-dir", payload.shadow_decoder_dir),
        ("--shadow-prior-dir", payload.shadow_prior_dir),
    )
    for flag, value in optional_pairs:
        if value:
            command.extend([flag, str(value)])
    return command


def _run_job(jobs_root: Path, experiments_root: Path, job_id: str) -> None:
    current = _read_json(_job_file(jobs_root, job_id))
    payload = AuditJobCreate.model_validate(current["payload"])
    workspace_path = experiments_root / payload.workspace_name
    command = _job_command(payload, workspace_path)
    _update_job(jobs_root, job_id, status="running", command=command)

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    summary_path = workspace_path / "summary.json"
    stdout_tail = completed.stdout.strip().splitlines()[-10:] if completed.stdout.strip() else []
    stderr_tail = completed.stderr.strip().splitlines()[-10:] if completed.stderr.strip() else []

    if completed.returncode == 0 and summary_path.exists():
        summary_payload = _read_json(summary_path)
        _update_job(
            jobs_root,
            job_id,
            status="completed",
            summary_path=str(summary_path),
            metrics=_extract_headline_metrics(summary_payload),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )
        return

    error_text = "\n".join(stderr_tail or stdout_tail) or "job execution failed"
    _update_job(
        jobs_root,
        job_id,
        status="failed",
        summary_path=str(summary_path) if summary_path.exists() else None,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=error_text,
    )


def create_app(
    *,
    experiments_root: str | Path | None = None,
    jobs_root: str | Path | None = None,
    auto_start_jobs: bool = False,
) -> FastAPI:
    if experiments_root is None:
        experiments_root = os.environ.get("DIFFAUDIT_EXPERIMENTS_ROOT", DEFAULT_EXPERIMENTS_ROOT)
    if jobs_root is None:
        jobs_root = os.environ.get("DIFFAUDIT_JOBS_ROOT", DEFAULT_JOBS_ROOT)
    experiments_path = Path(experiments_root).resolve()
    jobs_path = Path(jobs_root).resolve()
    experiments_path.mkdir(parents=True, exist_ok=True)
    jobs_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="DiffAudit Local API",
        version="0.1.0",
        description="Local API that exposes reproducible DiffAudit research workflows.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            experiments_root=str(experiments_path),
            jobs_root=str(jobs_path),
        )

    @app.get("/api/v1/models", response_model=list[ModelOption])
    def list_models() -> list[ModelOption]:
        return [ModelOption.model_validate(item) for item in MODEL_OPTIONS]

    @app.get("/api/v1/experiments/recon/best", response_model=ExperimentSummaryEnvelope)
    def get_best_recon_experiment() -> ExperimentSummaryEnvelope:
        try:
            best_summary_path = _best_recon_summary_path(experiments_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _summary_envelope(best_summary_path)

    @app.get(
        "/api/v1/experiments/{workspace_name}/summary",
        response_model=ExperimentSummaryEnvelope,
    )
    def get_workspace_summary(workspace_name: str) -> ExperimentSummaryEnvelope:
        if "/" in workspace_name or "\\" in workspace_name:
            raise HTTPException(status_code=400, detail="workspace_name must be a single directory")
        summary_path = experiments_path / workspace_name / "summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"workspace summary not found: {workspace_name}")
        return _summary_envelope(summary_path)

    @app.post("/api/v1/audit/jobs", response_model=AuditJobResponse, status_code=202)
    def create_audit_job(payload: AuditJobCreate) -> AuditJobResponse:
        record = _create_job_record(jobs_path, payload)
        if auto_start_jobs:
            thread = threading.Thread(
                target=_run_job,
                args=(jobs_path, experiments_path, str(record["job_id"])),
                daemon=True,
            )
            thread.start()
        return AuditJobResponse.model_validate(record)

    @app.get("/api/v1/audit/jobs/{job_id}", response_model=AuditJobResponse)
    def get_audit_job(job_id: str) -> AuditJobResponse:
        job_path = _job_file(jobs_path, job_id)
        if not job_path.exists():
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return AuditJobResponse.model_validate(_read_json(job_path))

    return app


app = create_app(auto_start_jobs=True)
