# 测试说明

这里放自动化测试。

## 当前范围

- `test_attack_registry.py`：CLI、planner、adapter、SecMI probe 与 smoke 相关测试
- `test_clid_smoke.py`：CLiD dry-run smoke 与 artifact summary 测试
- `test_recon_attack.py`：reconstruction-based 黑盒主线测试
- `test_local_api.py`：本地 HTTP API 契约、最佳证据查询与 job 持久化测试
- `test_variation_attack.py`：API-only variation 黑盒主线测试
- `test_smoke_pipeline.py`：配置加载、smoke pipeline 与资产说明测试
- `test_pia_adapter.py`：PIA runtime smoke 与 synthetic smoke 测试

## 运行建议

优先在 `diffaudit-research` 环境中运行：

```powershell
conda run -n diffaudit-research python -m unittest tests.test_attack_registry tests.test_smoke_pipeline
```

如果要跑全量：

```powershell
conda run -n diffaudit-research python -m unittest
```
