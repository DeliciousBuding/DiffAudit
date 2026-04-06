# 本地 API

这份文档说明如何在 `DiffAudit` 研究仓库内启动本地 HTTP API，并把现有 `recon` 黑盒主线暴露给平台或本机脚本调用。

## 目标

当前本地 API 的目标不是替代研究 CLI，而是为平台联调提供一个稳定、可查询、可提交受控任务的本地服务层。

原则：

- 优先复用 `experiments/*/summary.json`
- 优先复用现有 `python -m diffaudit ...` CLI
- 不复制实验大文件到平台仓库
- 路径型字段保留绝对路径，便于本机调试

## 启动方式

当前推荐入口是 Go 控制面：

```powershell
cd D:\Code\DiffAudit\Services\Local-API
go run ./cmd/local-api --host 127.0.0.1 --port 8765
```

覆盖本机目录：

```powershell
cd D:\Code\DiffAudit\Services\Local-API
go run ./cmd/local-api `
  --host 127.0.0.1 `
  --port 8765 `
  --experiments-root D:\Code\DiffAudit\Project\experiments `
  --jobs-root D:\Code\DiffAudit\Project\workspaces\local-api\jobs `
  --project-root D:\Code\DiffAudit\Project
```

Python 版 FastAPI 当前只保留为兼容入口，不再是推荐启动方式：

```powershell
conda run -n diffaudit-research python -m diffaudit serve-local-api `
  --host 127.0.0.1 `
  --port 8765 `
  --experiments-root D:\Code\DiffAudit\Project\experiments `
  --jobs-root D:\Code\DiffAudit\Project\workspaces\local-api\jobs
```

当前职责划分：

- Go：HTTP 控制面、job 持久化、目录扫描、子进程调度
- Python：继续执行 `recon_artifact_mainline` / `recon_runtime_mainline`

## 接口

### `GET /health`

返回服务状态和本机路径绑定：

```json
{
  "status": "ok",
  "experiments_root": "D:\\Code\\DiffAudit\\Project\\experiments",
  "jobs_root": "D:\\Code\\DiffAudit\\Project\\workspaces\\local-api\\jobs"
}
```

### `GET /api/v1/models`

返回当前本机已知的模型入口：

- `sd15-ddim`
- `kandinsky-v22`
- `dit-xl2-256`

### `GET /api/v1/experiments/recon/best`

返回 `recon` 当前最佳证据 summary。

优先读取：

- `experiments/blackbox-status/summary.json`

若该聚合文件缺失，则回退为扫描 `experiments/*/summary.json` 并按现有黑盒证据等级选择最佳 `recon` 记录。

当前默认会命中：

- `recon-runtime-mainline-ddim-public-25-step10`

### `GET /api/v1/experiments/{workspace}/summary`

读取某个实验目录下的 `summary.json`。

例如：

```powershell
curl http://127.0.0.1:8765/api/v1/experiments/recon-runtime-mainline-ddim-public-25-step10/summary
```

### `POST /api/v1/audit/jobs`

当前只允许受控的本地任务类型：

1. `recon_artifact_mainline`
2. `recon_runtime_mainline`

`workspace_name` 会映射到研究仓库下：

- `experiments/<workspace_name>/`

job 元数据会写到：

- `workspaces/local-api/jobs/<job_id>.json`

#### 提交 `recon_artifact_mainline`

```json
{
  "job_type": "recon_artifact_mainline",
  "workspace_name": "api-recon-artifact-mainline",
  "artifact_dir": "D:\\Code\\DiffAudit\\Project\\experiments\\recon-runtime-mainline-ddim-public-25-step10\\score-artifacts",
  "repo_root": "D:\\Code\\DiffAudit\\Project\\external\\Reconstruction-based-Attack",
  "method": "threshold"
}
```

#### 提交 `recon_runtime_mainline`

```json
{
  "job_type": "recon_runtime_mainline",
  "workspace_name": "api-recon-runtime-mainline",
  "target_member_dataset": "D:\\path\\target_member.pt",
  "target_nonmember_dataset": "D:\\path\\target_non_member.pt",
  "shadow_member_dataset": "D:\\path\\shadow_member.pt",
  "shadow_nonmember_dataset": "D:\\path\\shadow_non_member.pt",
  "target_model_dir": "D:\\path\\target_checkpoint",
  "shadow_model_dir": "D:\\path\\shadow_checkpoint",
  "repo_root": "D:\\Code\\DiffAudit\\Project\\external\\Reconstruction-based-Attack",
  "backend": "stable_diffusion",
  "scheduler": "ddim",
  "method": "threshold"
}
```

### `GET /api/v1/audit/jobs/{job_id}`

读取 job 当前状态。

### `GET /api/v1/audit/jobs`

按创建时间倒序返回当前已知 job 列表。

这个接口的用途是让平台或本机脚本在刷新页面后重新发现已提交任务，而不是只能依赖内存中的 `job_id`。

状态集合：

- `queued`
- `running`
- `completed`
- `failed`

任务完成后会补充：

- `summary_path`
- `metrics`
- `stdout_tail`
- `stderr_tail`

## 并发约束

当前 API 不允许对同一个 `workspace_name` 并发提交多个 `queued/running` 任务。

原因：

- 同一个 workspace 会落到同一实验目录
- 并发写入会互相覆盖 summary、artifact 和生成图片

如果同一个 workspace 已有活动任务，`POST /api/v1/audit/jobs` 会返回 `409 Conflict`。

## 平台联调建议

平台仓库不要复制研究结果，也不要自己再实现一套 `recon` 执行器。

平台只需要：

1. 调 `GET /api/v1/models` 获取可用模型入口
2. 调 `GET /api/v1/experiments/recon/best` 获取当前最佳证据
3. 调 `GET /api/v1/audit/jobs` 恢复本机已有任务列表
4. 调 `POST /api/v1/audit/jobs` 提交受控任务
5. 调 `GET /api/v1/audit/jobs/{job_id}` 轮询状态

## 当前边界

当前版本还没有做：

- 平台仓库内的反向代理
- 认证鉴权
- 多用户 job 隔离
- 任意 shell 命令执行
- `variation` / `CLiD` 的真实任务提交

当前版本已经打通：

- Go 本地 HTTP 服务入口
- `recon` 最佳证据查询
- 任意已知 workspace 的 summary 查询
- `recon_artifact_mainline` / `recon_runtime_mainline` 的受控 job 提交、列表与状态查询
- 同 workspace 活动任务的冲突保护

