# Multi-Reference Fusion Rollout Kickoff

更新时间: 2026-03-21

## 1. Branch And Environment

- 当前需求分支: `multiref-fusion-rollout`
- 基线分支: `backup/20260318-230150-current-changes`
- Python 环境: 项目根目录 `.venv`
- 本地环境要求: 非 Docker，本轮保留 Windows 本地开发约束说明

## 2. Confirmed Product Goal

把当前项目改造成正式支持以下能力:

- 多音色参考融合
- 多情感参考融合

要求:

- 两条变量轴分离，不互相污染实验结论
- 现有单参考调用继续可用
- 现有功能入口一并更新
- 必须补足单元测试和端到端测试

## 3. Current Technical Baseline

当前仓库已经具备:

- `indextts/fusion.py`: 融合 recipe 数据结构与基础工具
- `indextts/infer_v2.py`: 多参考融合底层逻辑
- `tools/*fusion*`: 实验、执行、评分脚本
- `docs/SPEAKER_FUSION_HANDOFF_20260319.md`: 实验与交接结论

当前仓库仍未完成正式 rollout 的主要缺口:

- `webui.py` 仍是单参考交互
- `indextts/cli.py` 仍未切到 `IndexTTS2` 多参考能力
- Python API 缺少正式默认方案的高层入口
- README / 中文文档未给出正式多参考用法
- 针对正式入口的单测与 e2e 还不够

## 4. Locked Experiment Conclusions

以下结论已经视为本轮正式默认方案输入，不再重新试验:

### 4.1 Timbre Multi-Reference

- default: `spk_cond_emb + speech_conditioning_latent`
- fallback: `speech_conditioning_latent`
- references: `A/B = 0.5 / 0.5`
- anchor: `symmetric`

### 4.2 Emotion Multi-Reference

- default: `emotion_tensor_anchor_a`
- fallback: `emotion_tensor_sym`
- references: `A/B = 0.5 / 0.5`
- anchor: `A`

### 4.3 Exclusion

以下路线不进入默认正式方案:

- `ref_mel`
- `style + ref_mel`

原因:

- 已在 CPU 复现短文本结构性失败
- 不是 GPU 噪声
- 只能保留为实验能力，不进入默认推荐

## 5. Active Assumptions

- 正式产品化优先围绕 `IndexTTS2`
- 多参考正式入口要优先封装“推荐方案”，而不是要求用户直接手写完整 `fusion_recipe`
- 高级用户仍可保留显式 `fusion_recipe` 覆盖能力
- WebUI / CLI / Python API 都要暴露推荐方案的使用路径
- e2e 以仓库内可自动化 smoke 为主，不要求真实大模型全量回归

## 6. Current Blockers

目前没有硬阻塞，主要是不确定项:

- Windows 本地流程可以写入文档和脚本约束，但当前会话无法直接做真实 Windows 运行验证
- WebUI 多参考交互需要在保持当前 UI 风格的前提下增加输入而不破坏现有流程

## 7. Next Intended Step

下一步不是直接编码，而是:

1. 创建 rollout 的 OpenSpec 变更
2. 写入 proposal / design / tasks
3. 把当前模式、假设、blocker、下一步同步到变更文档
4. 然后开始实现

