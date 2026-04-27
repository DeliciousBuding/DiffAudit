# Runtime（Legacy 文档）

这份文档保留旧时代的运行说明，供旧 trace 和历史交接参考。当前活跃执行层主称已经切换为 `Runtime`，入口位于 `Runtime-Server`（目录名待迁移）。

## 目标

当前本地 Runtime Server 的目标不是替代研究 CLI，而是为平台联调提供一个稳定、可查询、可提交受控任务的本地服务层。

原则：

- 优先复用 `experiments/*/summary.json` 与 `workspaces/*/artifacts/*.json`
- 优先复用现有 runner 与研究仓 admitted 产物
- 不复制实验大文件到平台仓库
- 路径型字段保留绝对路径，便于本机调试
- 平台优先读取 admitted 结果，不自己重拼黑盒/灰盒/白盒口径

当前 admitted 读源约束：

- 跨盒统一结果只认 `workspaces/implementation/artifacts/unified-attack-defense-table.json`
- `workspaces/intake/index.json` 只负责 promoted intake 合同，不等价于“所有 admitted 结果全集”
- `recon` 与 `DPDM` 当前仍需要结合冻结文档口径一起解释，不能只摘数值
- `PIA` 的 gray-box best summary 现在优先读取 intake manifest 中的 `canonical_summary / defense_summary`，只有 manifest 缺失时才回退目录扫描
- `PIA` 与 `GSA` 的 contract-specific best summary 都走 intake-first 读链；只有 intake 缺 summary 时才回退状态页或目录扫描
- `GSA` 当前 live intake 应指向 admitted `1k-3shadow` 资产根与对应 runtime mainline summary，而不是早期 CPU closed-loop summary

## 启动方式

当前推荐入口是 Go Runtime 控制面：

```powershell
cd <DIFFAUDIT_ROOT>/Runtime-Server
go run ./cmd/runtime --host 127.0.0.1 --port 8765
```

覆盖本机目录：

```powershell
cd <DIFFAUDIT_ROOT>/Runtime-Server
go run ./cmd/runtime `
  --host 127.0.0.1 `
  --port 8765 `
  --experiments-root <DIFFAUDIT_ROOT>/Research/experiments `
  --jobs-root <DIFFAUDIT_ROOT>/Research/workspaces/runtime/jobs `
  --project-root <DIFFAUDIT_ROOT>/Research
```

当前职责划分：

- Go：HTTP 控制面、job 持久化、目录扫描、子进程调度、admitted 结果读取
- bundled runners：继续执行 `recon` / `PIA` / `GSA` 的受控 job

## 接口

### `GET /health`

返回服务状态和本机路径绑定：

```json
{
  "status": "ok",
  "experiments_root": "<DIFFAUDIT_ROOT>/Research/experiments",
  "jobs_root": "<DIFFAUDIT_ROOT>/Research/workspaces/runtime/jobs"
}
```

### `GET /api/v1/models`

返回当前本机已知的模型入口：

- `sd15-ddim`
- `kandinsky-v22`
- `dit-xl2-256`

### `GET /api/v1/catalog`

返回当前 live contracts 与 admitted intake 信息。

当前可读到的关键信息包括：

- `contract_key`
- `track`
- `attack_family`
- `availability`
- `evidence_level`
- `admission_status`
- `admission_level`
- `provenance_status`
- `intake_manifest`
- `system_gap`

当 `research_root` 指向当前研究仓时，灰盒 `PIA` 的 intake 会带出：

- `admission_status = admitted`
- `admission_level = system-intake-ready`
- `provenance_status = workspace-verified`

`system_gap` 用来说明某条 live contract 目前还差什么，便于平台直接展示“还能做什么 / 还缺什么”，而不是重新硬编码解释。

边界说明：

- 当前 intake promoted 的重点是 `PIA` 与 `GSA` 资产合同。
- `recon` 与 `DPDM` 的 admitted 结果当前主要通过统一总表和冻结 workspace 文档暴露，而不是通过独立 intake manifest 暴露。

### `GET /api/v1/evidence/attack-defense-table`

返回研究仓当前 admitted attack-defense 总表。

当前读取：

- `workspaces/implementation/artifacts/unified-attack-defense-table.json`

不再作为系统权威读源：

- `workspaces/implementation/unified-attack-defense-table.json`

用途：

- 让平台或本机脚本直接读取 admitted main results
- 避免平台侧重复拼接 `recon / PIA / GSA / W-1` 的主口径

### `GET /api/v1/evidence/contracts/best?contract_key=...`

返回某条 live contract 当前最合适的 admitted summary。

当前优先级：

1. 先读 `workspaces/intake/index.json` 指向的 intake manifest
2. 再从 manifest 中优先读取 `canonical_summary / defense_summary`
3. 只有 intake 缺失或 manifest 不可用时，才回退状态页或目录扫描

当前最值得直接消费的 contract：

- `gray-box/pia/cifar10-ddpm`
- `white-box/gsa/ddpm-cifar10`

当前期望命中的 admitted summary：

- `PIA` -> `pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive`
- `GSA` -> `gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1`

边界说明：

- `recon` 当前仍保留 `GET /api/v1/experiments/recon/best` 这个黑盒便捷入口
- `recon` 与 `DPDM` 仍需要结合冻结 workspace 文档解释，不应只摘 summary 数值

### `GET /api/v1/experiments/recon/best`

返回 `recon` 当前最佳证据 summary。

优先读取：

- `experiments/blackbox-status/summary.json`

若该聚合文件缺失，则回退为扫描 `experiments/*/summary.json` 并按现有黑盒证据等级选择最佳 `recon` 记录。

当前默认会命中：

- `recon-runtime-mainline-ddim-public-100-step30`

### `GET /api/v1/experiments/{workspace}/summary`

读取某个实验目录下的 `summary.json`。

例如：

```powershell
curl http://127.0.0.1:8765/api/v1/experiments/recon-runtime-mainline-ddim-public-100-step30/summary
```

### `POST /api/v1/audit/jobs`

当前允许的 live job 类型由 `contract_key + job_type` 决定。

当前 live contracts：

1. `black-box/recon/sd15-ddim`
2. `gray-box/pia/cifar10-ddpm`
3. `white-box/gsa/ddpm-cifar10`

`workspace_name` 会映射到研究仓或本地 Runtime Server 工作目录下对应的运行产物目录。

job 元数据会写到：

- `workspaces/local-api/jobs/<job_id>.json`

#### 提交 `recon_artifact_mainline`

```json
{
  "job_type": "recon_artifact_mainline",
  "contract_key": "black-box/recon/sd15-ddim",
  "workspace_name": "api-recon-artifact-mainline",
  "artifact_dir": "<DIFFAUDIT_ROOT>/Research/experiments/recon-runtime-mainline-ddim-public-100-step30/score-artifacts",
  "repo_root": "<DIFFAUDIT_ROOT>/Research/external/Reconstruction-based-Attack",
  "method": "threshold"
}
```

#### 提交 `recon_runtime_mainline`

```json
{
  "job_type": "recon_runtime_mainline",
  "contract_key": "black-box/recon/sd15-ddim",
  "workspace_name": "api-recon-runtime-mainline",
  "target_member_dataset": "D:\\path\\target_member.pt",
  "target_nonmember_dataset": "D:\\path\\target_non_member.pt",
  "shadow_member_dataset": "D:\\path\\shadow_member.pt",
  "shadow_nonmember_dataset": "D:\\path\\shadow_non_member.pt",
  "target_model_dir": "D:\\path\\target_checkpoint",
  "shadow_model_dir": "D:\\path\\shadow_checkpoint",
  "repo_root": "<DIFFAUDIT_ROOT>/Research/external/Reconstruction-based-Attack",
  "backend": "stable_diffusion",
  "scheduler": "ddim",
  "method": "threshold"
}
```

#### 提交 `pia_runtime_mainline`

```json
{
  "job_type": "pia_runtime_mainline",
  "contract_key": "gray-box/pia/cifar10-ddpm",
  "workspace_name": "api-pia-runtime-mainline-001",
  "runtime_profile": "docker-default",
  "repo_root": "<DIFFAUDIT_ROOT>/Research/external/PIA",
  "assets": {
    "member_split_root": "<DIFFAUDIT_ROOT>/Research/external/PIA/DDPM"
  },
  "job_inputs": {
    "config": "<DIFFAUDIT_ROOT>/Research/tmp/configs/pia-cifar10-graybox-assets.local.yaml",
    "device": "cpu",
    "num_samples": "16"
  }
}
```

#### 提交 `gsa_runtime_mainline`

```json
{
  "job_type": "gsa_runtime_mainline",
  "contract_key": "white-box/gsa/ddpm-cifar10",
  "workspace_name": "api-gsa-runtime-mainline-001",
  "runtime_profile": "docker-default",
  "repo_root": "<DIFFAUDIT_ROOT>/Research/workspaces/white-box/external/GSA",
  "assets": {
    "assets_root": "<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1"
  },
  "job_inputs": {
    "resolution": "32",
    "ddpm_num_steps": "20",
    "sampling_frequency": "2",
    "attack_method": "1"
  }
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

平台仓库不要复制研究结果，也不要自己再实现一套攻击执行器或结果聚合器。

平台只需要：

1. 先调 `GET /api/v1/evidence/attack-defense-table` 读取 admitted 统一主结果
2. 再调 `GET /api/v1/evidence/contracts/best?contract_key=gray-box/pia/cifar10-ddpm` 读取 `PIA` 当前最佳 admitted summary
3. 调 `GET /api/v1/catalog` 获取 live contract、intake 状态与 `system_gap`
4. 调 `GET /api/v1/models` 获取可用模型入口
5. 在需要黑盒单线详情时调 `GET /api/v1/experiments/recon/best`
6. 调 `GET /api/v1/audit/jobs` 恢复本机已有任务列表
7. 调 `POST /api/v1/audit/jobs` 提交受控任务
8. 调 `GET /api/v1/audit/jobs/{job_id}` 轮询状态

## 当前边界

当前版本还没有做：

- 平台仓库内的反向代理
- 认证鉴权
- 多用户 job 隔离
- 任意 shell 命令执行
- `variation` / `CLiD` 的真实任务提交

当前版本已经打通：

- Go 本地 HTTP 服务入口
- `catalog` admitted intake 查询
- admitted attack-defense 总表查询
- `PIA` / `GSA` contract-specific best summary 的 intake-first 查询
- `recon` 最佳证据查询
- 任意已知 workspace 的 summary 查询
- `recon` / `PIA` / `GSA` live job 的受控提交、列表与状态查询
- 同 workspace 活动任务的冲突保护
