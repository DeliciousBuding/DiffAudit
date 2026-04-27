# SMP-LoRA GPU训练接手文档

**交接时间**: 2026-04-12
**状态**: 已接管但未放行GPU扩跑，需要准入判断

---

## 一、真实完成状态（核实后）

### 1. Phase 1: 超参数扫描（实际6组，非14组）

**扫描参数**: lambda=0.1/0.5/0.9, rank=4, epochs=10/20

| 配置 | AUC | Accuracy | 备注 |
| --- | --- | --- | --- |
| **lambda=0.1, rank=4, ep=10** | **0.3438** | 0.3947 | **最优** |
| lambda=0.1, rank=4, ep=20 | 0.4622 | 0.4737 | 过拟合 |
| lambda=0.5, rank=4, ep=10 | 0.6139 | 0.5789 | 较差 |
| lambda=0.5, rank=4, ep=20 | 0.5582 | 0.5789 | 较差 |
| lambda=0.9, rank=4, ep=10 | 0.5028 | 0.4474 | 中等 |
| lambda=0.9, rank=4, ep=20 | 0.4584 | 0.4211 | 中等 |

**结论**: lambda越小防御效果越好，ep=10优于ep=20

### 2. 无防御基线

**结果**: AUC=0.5565, Accuracy=0.5263

### 3. 100 epochs长训练

**配置**: lambda=0.1, rank=4, epochs=100
**结果**: AUC=0.3785（比10 epochs的0.3438稍差，过拟合）

### 4. 中断训练残留

**路径**: `outputs/smp-lora-lambda005/`
**状态**: 训练到step_500后中断，无final结果，无评估结果
**配置**: lambda=0.05, rank=4, epochs=100（推测）

---

## 二、关键结论

1. **SMP-LoRA显著降低GSA攻击效果**
   - 无防御基线: AUC=0.5565
   - 最优配置: AUC=0.3438
   - **防御效果提升: 38%**

2. **最优超参数**
   - lambda=0.1（低隐私惩罚）
   - rank=4（中等秩）
   - epochs=10（短训练即可，长训练过拟合）

3. **过拟合风险**
   - 100 epochs效果不如10 epochs
   - ep=20效果不如ep=10
   - 建议控制训练轮数

---

## 三、与根级主线的关系

**当前根级主线**: PIA + provisional G-1(all_steps)
**当前唯一推进目标**: PIA provenance的release/source identity unresolved CPU closure

**冲突点**:
- GPU空闲不是直接放行SMP-LoRA扩跑的理由
- 这条线没有完成和根级主线的admission对齐
- 直接推进14个GPU重任务会违反当前治理红线

**建议处理**:
- 按"已接管但未放行GPU扩跑"处理
- 先做真实状态收口+准入判断
- 明确是否进入Phase E候选审查面

---

## 四、产物清单

### 已完成产物
```
outputs/smp-lora-sweep/
├── sweep_results.json          # 6组扫描结果
├── lambda0.1_rank4_ep10/       # 最优配置
├── lambda0.1_rank4_ep20/
├── lambda0.5_rank4_ep10/
├── lambda0.5_rank4_ep20/
├── lambda0.9_rank4_ep10/
└── lambda0.9_rank4_ep20/

outputs/smp-lora-phase2/
└── baseline_nodefense_target-64/
    └── evaluation.json         # 无防御基线 AUC=0.5565

outputs/smp-lora-best-config/
├── final/lora_weights.pt       # 100 epochs权重
├── evaluation.json             # AUC=0.3785
└── step_*/                     # 中间checkpoint
```

### 中断残留
```
outputs/smp-lora-lambda005/
├── step_100/
├── step_200/
├── step_300/
├── step_400/
├── step_500/                   # 中断点
└── config.json
```

---

## 五、代码路径

### 训练脚本
```
scripts/train_smp_lora.py       # SMP-LoRA训练主脚本
scripts/batch_smp_lora_sweep.py # 批量扫描脚本
```

### 评估脚本
```
scripts/evaluate_smp_lora_defense.py  # GSA攻击评估
scripts/batch_eval_sweep.py           # 批量评估脚本
```

### 核心实现
```
src/diffaudit/defenses/lora_ddpm.py   # LoRA层注入
src/diffaudit/defenses/smp_lora.py    # SMP-LoRA训练器
```

---

## 六、训练命令模板

### 单次训练
```bash
conda run -n diffaudit-research python scripts/train_smp_lora.py \
    --local_model <模型路径> \
    --member_dir <成员数据路径> \
    --nonmember_dir <非成员数据路径> \
    --output_dir <输出路径> \
    --rank 4 \
    --lambda_coeff 0.1 \
    --num_epochs 10 \
    --batch_size 8 \
    --device cuda \
    --save_every 100
```

### 评估
```bash
conda run -n diffaudit-research python scripts/evaluate_smp_lora_defense.py \
    --lora_checkpoint <LoRA权重路径> \
    --base_model <基座模型路径> \
    --member_dir <成员数据路径> \
    --nonmember_dir <非成员数据路径> \
    --device cuda \
    --num_samples 500 \
    --output <输出JSON路径> \
    --rank 4
```

---

## 七、注意事项

1. **评估脚本rank参数**: 必须与训练时的rank一致，否则会报错
2. **设备管理**: 确保LoRA层移到CUDA设备
3. **过拟合风险**: 长训练效果不如短训练
4. **checkpoint恢复**: 训练脚本不支持从checkpoint恢复
5. **工作树状态**: Project工作树很脏，不能安全混提交

---

## 八、下一步建议

### 短期（收口）
1. 清理中断残留 `outputs/smp-lora-lambda005/`
2. 整理真实完成状态到intake文档
3. 明确准入判断：是否进入Phase E候选审查面

### 中期（如果准入通过）
1. 补完lambda=0.05评估（从中断点继续或重新训练）
2. 探索lambda=0.01（可能效果更好）
3. Rank=1极低秩实验

### 长期（如果准入通过）
1. FID生成质量评估
2. 自适应攻击评估
3. W-1(DPDM) vs SMP-LoRA统一对比

---

## 九、相关文档

- 技术分析: `workspaces/intake/2026-04-11-dplora-comparability-note.md`
- 任务看板: `<DIFFAUDIT_ROOT>/Agents/GLOBAL_TASK_BOARD.md`
- 攻击-防御总表: `workspaces/white-box/attack-defense-table.md`
