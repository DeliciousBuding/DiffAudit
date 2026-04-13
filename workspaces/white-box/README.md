# 白盒工作区

用于存放：

- 白盒论文阅读笔记
- 白盒攻击假设
- 内部信号利用方案
- 白盒方向任务认领

## 当前建议任务

1. 当前白盒主线是 `GSA + W-1`，当前目标不是再刷 attack，而是把 admitted `GSA rerun1` 与 `W-1 strong-v3 full-scale` 推进到 same-protocol bridge
2. 当前白盒防御优先做 `W-1`，即把 `external/DPDM` 接成正式 comparator，并补齐 portable 启动入口
3. 当前 admitted `GSA` 资产根目录与交接口径统一看 `assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/manifests/cifar10-ddpm-1k-3shadow-epoch300-rerun1.json`；`assets/gsa/*` 只保留为早期 closed-loop legacy 根
4. 当前白盒主合同文档看 `2026-04-09-whitebox-same-protocol-bridge.md`
5. 同一时段只允许一个白盒主 GPU 问题；当前问题是 same-protocol bridge，不并行新开第二条白盒长任务
6. 综合进度口径统一看 `../../docs/comprehensive-progress.md`
