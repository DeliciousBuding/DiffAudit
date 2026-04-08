# 本地 API

这份文档说明 `Project` 如何与 `Services/Local-API` 对接。

当前口径已经不是“给 `recon` 暴露一个本地 HTTP 壳”，而是：

- `Project` 继续作为研究与证据源
- `Services/Local-API` 作为统一控制面与 admitted evidence 读链
- 黑盒 / 灰盒 / 白盒共用同一套 `catalog / evidence / audit job` 合同

## 当前定位

`Local-API` 现在负责三件事：

1. 暴露当前 admitted 的研究结果
2. 提供受控的本地任务提交入口
3. 为 `Platform` 或本机脚本提供稳定的 HTTP 合同

它不替代研究 CLI，也不复制 `Project` 里的大资产。

## 启动方式

当前推荐入口是 Go 控制面：

```powershell
cd D:\Code\DiffAudit\Services\Local-API
powershell -ExecutionPolicy Bypass -File .\run-local-api.ps1
```

如需显式指定路径：

```powershell
cd D:\Code\DiffAudit\Services\Local-API
go run ./cmd/local-api `
  --host 127.0.0.1 `
  --port 8765 `
  --project-root D:\Code\DiffAudit\Project `
  --experiments-root D:\Code\DiffAudit\Project\experiments `
  --jobs-root D:\Code\DiffAudit\Project\workspaces\local-api\jobs `
  --repo-root D:\Code\DiffAudit\Project `
  --execution-mode local
```

## 当前读接口

### `GET /health`

返回服务状态与绑定路径。

### `GET /diagnostics`

返回当前执行模式、关键根路径、runner 存在性和 Docker / 调度器配置。

### `GET /api/v1/catalog`

返回当前 live contract 列表，并在 `project_root` 可用时自动补 intake 元数据。

当前已 admitted / live 的核心合同：

- `black-box/recon/sd15-ddim`
- `gray-box/pia/cifar10-ddpm`
- `white-box/gsa/ddpm-cifar10`

当前 intake 里已经明确可被系统读取的元数据包括：

- `admission_status`
- `admission_level`
- `provenance_status`
- `intake_manifest`

其中：

- `PIA` 当前 `provenance_status = workspace-verified`
- `GSA` 当前 `provenance_status = workspace-verified`

### `GET /api/v1/evidence/attack-defense-table`

返回当前统一 attack-defense 总表。

读取来源：

- `workspaces/implementation/artifacts/unified-attack-defense-table.json`

当前用途：

- 让平台直接消费 admitted 的黑盒 / 灰盒 / 白盒主结果
- 避免平台侧自己再拼 `recon / PIA / GSA / W-1` 主口径

### `GET /api/v1/evidence/contracts/best?contract_key=...`

返回指定 live contract 的最佳 admitted summary envelope。

示例：

```powershell
curl "http://127.0.0.1:8765/api/v1/evidence/contracts/best?contract_key=gray-box/pia/cifar10-ddpm"
curl "http://127.0.0.1:8765/api/v1/evidence/contracts/best?contract_key=white-box/gsa/ddpm-cifar10"
```

当前意义：

- 去掉平台对 `recon` 专属接口的依赖
- 让灰盒 / 白盒也能按合同直接读取最佳 admitted 结果

### `GET /api/v1/experiments/recon/best`

这是保留的兼容接口。

它本质上等价于：

```text
GET /api/v1/evidence/contracts/best?contract_key=black-box/recon/sd15-ddim
```

### `GET /api/v1/experiments/{workspace}/summary`

读取指定 workspace 的 `summary.json`。

适合：

- 追具体 run
- 调试某次实验目录
- 平台进入单次结果详情页

## 当前写接口

### `POST /api/v1/audit/jobs`

当前受控 job 仍以 admitted runner 为准，必须带：

- `job_type`
- `contract_key`
- `workspace_name`

当前 live job 主要包括：

- `recon_artifact_mainline`
- `recon_runtime_mainline`
- `pia_runtime_mainline`
- `gsa_runtime_mainline`

示例：灰盒 `PIA`

```json
{
  "job_type": "pia_runtime_mainline",
  "contract_key": "gray-box/pia/cifar10-ddpm",
  "workspace_name": "api-pia-runtime-mainline-001",
  "runtime_profile": "docker-default",
  "repo_root": "D:/Code/DiffAudit/Project/external/PIA",
  "job_inputs": {
    "config": "D:/Code/DiffAudit/Project/tmp/configs/pia-cifar10-graybox-assets.local.yaml",
    "device": "cpu",
    "num_samples": "16",
    "provenance_status": "workspace-verified"
  }
}
```

示例：白盒 `GSA`

```json
{
  "job_type": "gsa_runtime_mainline",
  "contract_key": "white-box/gsa/ddpm-cifar10",
  "workspace_name": "api-gsa-runtime-mainline-001",
  "runtime_profile": "docker-default",
  "repo_root": "D:/Code/DiffAudit/Project/workspaces/white-box/external/GSA",
  "job_inputs": {
    "assets_root": "D:/Code/DiffAudit/Project/workspaces/white-box/assets/gsa",
    "resolution": "32",
    "ddpm_num_steps": "20",
    "sampling_frequency": "2",
    "attack_method": "1",
    "provenance_status": "workspace-verified"
  }
}
```

### `GET /api/v1/audit/jobs`

返回已知任务列表。

### `GET /api/v1/audit/jobs/{job_id}`

返回单个任务状态、命令、输出尾部和 `summary_path`。

## 当前平台接法

`Platform` 当前应该优先走这条链：

1. `GET /api/v1/catalog`
2. `GET /api/v1/evidence/attack-defense-table`
3. `GET /api/v1/evidence/contracts/best?contract_key=...`
4. `GET /api/v1/audit/jobs`
5. `POST /api/v1/audit/jobs`
6. `GET /api/v1/audit/jobs/{job_id}`

不建议再继续围绕 `recon` 单独做页面结构或接口假设。

## 当前边界

当前已经打通：

- 三盒 live contract 的统一 catalog
- admitted attack-defense 总表读取
- 合同级最佳证据摘要读取
- `recon / PIA / GSA` 的受控任务提交骨架

当前仍未打通：

- 黑盒防御正式 admitted 结果
- `variation / Towards` 的真实 API 资产闭环
- `SecMI` 的真实资产闭环
- 多用户鉴权与隔离
