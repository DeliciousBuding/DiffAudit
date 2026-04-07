# `src/diffaudit/attacks`

这里按方法线放攻击相关逻辑。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `secmi.py` | `SecMI` 的 planner、资产探测、workspace 校验 |
| `secmi_adapter.py` | `SecMI` 的 adapter、runtime probe、synthetic smoke |
| `pia.py` | `PIA` 的 planner、资产探测、workspace 校验 |
| `pia_adapter.py` | `PIA` 的 runtime probe、runtime smoke、synthetic smoke |
| `clid.py` | `CLiD` 的 planner、asset probe、dry-run、artifact summary |
| `recon.py` | reconstruction-based 纯黑盒主线 |
| `variation.py` | API-only black-box variation 主线 |
| `registry.py` | 统一 planner 注册表 |

## 当前方法分层

### 黑盒

- `clid`
- `recon`
- `variation`

### 灰盒

- `secmi`
- `pia`

### 白盒

- `gsa.py`
  - `GSA` 的 workspace 校验、资产探针和 real-asset closed-loop mainline

## 推荐新增方式

新增一个方法线时，建议按这个顺序落：

1. 新建 `<method>.py`
2. 在 `registry.py` 注册 planner
3. 在 `cli.py` 接入 `plan-*`
4. 补 `probe-*` 和 `dry-run-*`
5. 视情况增加 `runtime smoke` 或 `artifact summary`
6. 同步更新测试和 README
