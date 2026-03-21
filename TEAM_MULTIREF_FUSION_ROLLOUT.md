# Multi-Reference Fusion Rollout Team Charter

更新时间: 2026-03-21

## 1. Objective

本轮目标是把 `IndexTTS2` 现有的实验性多参考融合能力收敛成正式可用功能，并覆盖当前项目的主要使用面:

- Python API
- WebUI
- CLI
- README / 中文文档
- 单元测试
- 端到端测试

功能范围必须同时包含两条彼此分离的变量轴:

- 多音色参考融合
- 多情感参考融合

本轮默认方案以已完成实验结论为准:

- timbre default: `spk_cond_emb + speech_conditioning_latent`
- timbre fallback: `speech_conditioning_latent`
- emotion default: `emotion_tensor_anchor_a`
- emotion fallback: `emotion_tensor_sym`

## 2. Active Mode

- 协作模式: `parallel`
- 负责人: `lead / integration`
- 停止条件:
  - 正式功能可用
  - 现有单参考路径保持兼容
  - 单测与端到端测试通过
  - 文档与 OpenSpec 变更同步完成

## 3. Team Roles

### 3.1 Design

负责人: `design`

职责:

- 把实验结论转成正式接口与默认行为
- 明确 timbre / emotion 的输入模型和变量边界
- 定义 WebUI / CLI / Python API 的用户路径
- 维护 rollout 的 OpenSpec `proposal.md` 和 `design.md`

交付:

- 正式接口约束
- 默认方案与回退方案
- 回滚策略
- 文档入口更新范围

### 3.2 Development

负责人: `development`

职责:

- 实现 Python API、WebUI、CLI、文档示例和测试改造
- 保证 `.venv` 为唯一开发环境
- 控制并行改动边界，避免多人同时写同一文件
- 维护 rollout 的 OpenSpec `tasks.md`

交付:

- 可运行代码
- 单元测试
- 端到端 smoke / e2e 覆盖
- 与默认方案一致的用户入口

### 3.3 Review

负责人: `review`

职责:

- 审查行为回归、接口歧义、测试缺口和兼容性风险
- 检查文档与实现是否一致
- 检查是否保留对单参考旧调用的兼容
- 审查每轮迭代的提交范围与验收证据

交付:

- 风险清单
- 测试门槛
- 合并前 review 结论

## 4. Ownership Boundaries

为避免并行冲突，本轮按以下边界拆分:

- `indextts/fusion.py`, OpenSpec 设计文档: `design`
- `indextts/infer_v2.py`, `indextts/cli.py`, `webui.py`: `development`
- `tests/*`, README, 验证结论、回归检查: `review` 提需求，`development` 落地

若某一轮出现同文件交叉改动，立即降级为 `delegate` 或 `single-writer`。

## 5. Handoff Protocol

每个工作周期必须包含三方参与:

1. `design` 先定义本轮目标、接口边界、验收口径
2. `development` 按边界实现并记录变更
3. `review` 基于代码和测试结果给出结论

信息传达要求:

- 所有关键决策先写入文档再继续实现
- 每个新文档独立 commit
- compact 后一律从文档恢复，不依赖隐藏上下文
- 每轮结束都要更新:
  - 当前模式
  - 当前假设
  - 当前 blocker
  - 下一步动作

## 6. Recovery Sources

发生 compact 或会话中断时，按以下顺序恢复:

1. `TEAM_MULTIREF_FUSION_ROLLOUT.md`
2. `MULTIREF_FUSION_KICKOFF_20260321.md`
3. rollout 对应的 `openspec/changes/<change>/`
4. `docs/SPEAKER_FUSION_HANDOFF_20260319.md`

## 7. Local Development Constraints

- 开发环境使用项目根目录下 `.venv`
- 不使用 Docker
- 本轮文档、脚本和本地开发辅助内容直接放项目根目录或现有项目目录，不新增额外外层工作区
- 目标兼容本地 Windows 开发流程，但当前验证环境以仓库内现有 Linux 工作区为准

## 8. Initial Acceptance Gate

开始编码前必须满足:

- 新需求分支已创建
- 团队编排文档已提交
- 当前阶段快照已提交
- rollout OpenSpec 变更已创建并写入当前模式 / 假设 / blocker / 下一步

