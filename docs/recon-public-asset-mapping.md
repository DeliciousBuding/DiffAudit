# Recon Public Asset Mapping

这份说明用于记录公开 `recon` 资产包的当前最可辩护语义映射，避免后续把文件名直接当成已经确认的论文真值。

公开资产包来源：

- DOI: `10.5281/zenodo.13371475`

相关本地根目录：

- `external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/`

当前派生映射注释还存在于：

- `external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-public-10/mapping-note.md`
- `external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-public-25/mapping-note.md`
- `external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-public-50/mapping-note.md`
- `external/recon-assets/ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models/derived-public-100/mapping-note.md`

## 当前最稳妥的映射

- `target_member`
  - `source-datasets/partial-100-target/member/dataset.pkl`
- `target_non_member`
  - `source-datasets/partial-100-target/non_member/dataset.pkl`
- `shadow_non_member`
  - `source-datasets/100-shadow/non_member/dataset.pkl`
- `shadow_member`
  - 当前只能临时用 `source-datasets/100-target/non_member/dataset.pkl` 作为代理

## 为什么这样映射

第一，`partial-100-target` 是当前公开数据里唯一同时显式给出 `member/` 与 `non_member/` 两个子目录的 target split，因此它是唯一能直接支撑 target-member / target-non-member 二元划分的来源。

第二，`100-shadow` 目录目前只看到 `non_member/`，没有同层级的 `member/`，因此它最多只能直接承担 `shadow_non_member`。

第三，`100-target/non_member` 在命名上不是一个干净的 `shadow_member`，只是当前公开包里最接近可用的影子正类代理。把它当成 `shadow_member` 是工程占位，而不是已经核准的论文语义。

第四，checkpoint 命名本身支持“target 线 / shadow 线 + partial/full”两条轴：

- `celeba_partial_target`
- `celeba_target`
- `celeba_partial_shadow`
- `celeba_shadow`

但公开数据集目录没有给出与这些 checkpoint 完全一一对应的四象限 split，所以当前只能做最保守映射，而不能声称“target/shadow/member/non-member 四元语义已完全核准”。

第五，`derived-public-*` 目录下的 `mapping-note.md` 已经把当前运行链实际使用的映射显式记录成：

- `target_member`
- `target_non_member`
- `shadow_non_member`
- `shadow_member_proxy`

这说明当前语义已经不是“临时口头约定”，而是本地派生资产的一部分；但因为它写的仍是 `shadow_member_proxy`，所以它仍然只能支撑“受约束的本地语义”，不能直接升级成论文完全核准。

## 当前结论

- 可以合理声称：公开资产已经足以支撑 `Stable Diffusion + DDIM` 的 `10-sample public runtime-mainline`，以及 `kandinsky_v22` 的最小真实 `runtime-mainline`
- 可以合理声称：当前 `derived-public-{10,25,50,100}` 都带有一致的本地映射注释，因此现有 `recon` 主证据链是“语义受约束但本地自洽”的
- 不能合理声称：当前公开资产已经严格对齐论文里的完整 target/shadow/member/non-member 语义
- issue #10 后，strict paper-faithful `Attack-I` 入口必须使用 `check-recon-stage0-paper-gate`；当前 gate 的正确结果是 `blocked / paper_aligned_semantics = false`

## 推荐执行策略

1. `1-sample` 与更小规模 smoke 可继续用当前映射推进，目的在于验证系统与运行链。
2. 扩到更大样本规模前，先在报告中显式注明 `shadow_member` 仍是代理语义。
3. 若目标是 strict paper-faithful `Attack-I`，先运行 `check-recon-stage0-paper-gate`；在该 gate 返回 `blocked` 时，不允许把结果写成 paper-aligned。
4. 如果后续拿到更完整的 split 说明或补充资产，再更新本文件，不要直接在命令里静默替换映射。
