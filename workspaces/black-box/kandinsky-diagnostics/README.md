# Kandinsky Diagnostics

这个目录收纳 `Kandinsky v2.2` 运行异常的临时诊断脚本与最小说明。

## 当前内容

- `kandinsky_stage_probe.py`
  - 用于把 `image_encoder` / `unet` / `prior` / `decoder` / lora 加载与前向阶段分段计时

## 作用边界

- 这里只放临时诊断工具
- 不作为正式实验入口
- 若后续 `Kandinsky 10/10` 恢复推进，应优先先跑这里的阶段诊断，再决定是否继续占 GPU
