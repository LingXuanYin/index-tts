# Speaker Fusion 工作区迁移交接

更新时间: 2026-03-19

## 1. 本轮范围

本轮工作聚焦在 IndexTTS2 的多参考说话人融合能力，以及配套的实验清单生成、结果元数据落盘和实验脚本。目标是把“可在推理期完成的 speaker fusion”先落成可运行代码和可扩展实验框架，再为后续十批次实验和默认方案筛选留出接口。

本轮没有把数据集、实验产物和本地辅助目录提交到远端仓库。迁移工作区时，这些内容需要单独迁移。

## 2. 当前仓库定位

- 当前分支: `backup/20260318-230150-current-changes`
- 当前提交: `e44717462b30b0ee3b34b3ef9f72c9173932c416`
- 提交标题: `feat: add speaker fusion changes without datasets`
- 远端仓库: `https://github.com/LingXuanYin/index-tts.git`
- PR 入口: `https://github.com/LingXuanYin/index-tts/pull/new/backup/20260318-230150-current-changes`

截至本次交接，工作区是干净状态，没有未提交改动。

## 3. 已完成成果

### 3.1 推理侧 speaker fusion 能力已接入

核心实现已经接入 [`indextts/infer_v2.py`](../indextts/infer_v2.py) 和 [`indextts/fusion.py`](../indextts/fusion.py)，包括:

- 支持多参考 `fusion_recipe`
- 支持层级:
  - `spk_cond_emb`
  - `speech_conditioning_latent`
  - `prompt_condition`
  - `style`
  - `ref_mel`
- 支持 branch 级别分配:
  - `speaker`
  - `prompt`
  - `style`
  - `ref_mel`
  - `emotion`
- 支持 metadata 输出:
  - 每次合成可记录 recipe、branch、timing、segment 信息
- 支持融合相关缓存键:
  - 不同 recipe 不会误复用缓存

### 3.2 实验层能力已接入

核心实验脚本已落在 [`tools/speaker_fusion_experiment.py`](../tools/speaker_fusion_experiment.py)，能力包括:

- ten-batch slice 清单生成
- 全组合 manifest 生成
- baseline case 与 fusion case 混排
- JSONL / CSV manifest 读写
- 已完成 case 自动跳过，支持断点续跑
- 支持 branch profile 组合
- 结果 metadata 与运行状态回写

### 3.3 Experimental 层已隔离

以下能力已被归类为 experimental，默认路径不会自动启用:

- waveform mixing
- `cat_condition`
- `vc_target`

这部分逻辑已经进入推理代码，但需要显式启用，不会污染默认单说话人路径。

### 3.4 OpenSpec 文档已补齐

已补齐本次变更的 OpenSpec 设计与规格:

- [`openspec/changes/add-speaker-fusion-to-indextts-v2/proposal.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/proposal.md)
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/design.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/design.md)
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-inference/spec.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-inference/spec.md)
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-experiments/spec.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-experiments/spec.md)
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-evaluation/spec.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/specs/speaker-fusion-evaluation/spec.md)
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/tasks.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/tasks.md)

### 3.5 已有验证结果

已执行并通过:

```bash
pytest tests/test_speaker_fusion_experiment.py -q
```

结果:

- `5 passed in 0.05s`

对应测试文件:

- [`tests/test_speaker_fusion_experiment.py`](../tests/test_speaker_fusion_experiment.py)

## 4. 本轮提交涉及的关键文件

新增:

- [`indextts/fusion.py`](../indextts/fusion.py)
- [`tools/speaker_fusion_experiment.py`](../tools/speaker_fusion_experiment.py)
- [`tests/test_speaker_fusion_experiment.py`](../tests/test_speaker_fusion_experiment.py)
- `openspec/changes/add-speaker-fusion-to-indextts-v2/*`

修改:

- [`indextts/infer_v2.py`](../indextts/infer_v2.py)
- [`indextts/gpt/model_v2.py`](../indextts/gpt/model_v2.py)
- [`indextts/s2mel/modules/bigvgan/utils.py`](../indextts/s2mel/modules/bigvgan/utils.py)
- [`.gitignore`](../.gitignore)

当前 `.gitignore` 已新增忽略规则，避免再次把本地数据和实验产物误提交:

- `/artifacts/`
- `/data/`
- `/.codex/`

## 5. 未提交到远端的本地内容

以下内容没有进入当前远端分支，迁移工作区时如果后续还需要继续实验，必须手动迁移:

- `data/`
- `artifacts/`
- `checkpoints/` 下除 `config.yaml` 之外的模型权重
- `.venv/`
- `.uv-cache/`
- `.pytest_cache/`

说明:

- 本地确实生成过 speaker fusion 相关实验产物与 open-source dataset 目录
- 这些内容未进 Git，是刻意处理，原因是体积过大且会阻塞远端推送

## 6. 剩余问题

### 6.1 十批次全组合实验还没有在当前提交内闭环

OpenSpec 中以下任务仍未完成:

- `5.2 Run the first ten-batch open-source full-combination experiment`
- `5.3 Document ... and the selected default recommendation`

当前状态是:

- 实验脚本和 manifest 生成逻辑已完成
- 本地数据切片和实验产物曾存在，但未纳入当前远端分支
- 当前提交里没有可直接复现的 ten-batch 结果文件
- 还没有基于完整实验结果选出默认推荐 fusion scheme

### 6.2 默认推荐方案尚未确定

目前只有框架与打分入口，没有最终结论:

- 哪个 supported-tier 组合作为默认方案
- 对 sequence tensor 应优先使用哪种 anchor 模式
- experimental 层是否值得进入后续推荐集

这部分仍需要在迁移后的环境里重跑 ten-batch 实验，再做决策。

### 6.3 端到端真实推理尚未在本轮重新回归

本轮完成了单元测试与文档/脚本落地，但没有在当前提交上重新做一轮“带本地 checkpoints 的完整 speaker fusion 端到端回归”。

因此还没有在交接文档里宣称以下事项已完成验证:

- 真实模型权重下的多参考融合音频质量
- `metadata_output_path` 在真实长句场景下的稳定性
- experimental 层在真实批量生成时的退化边界

### 6.4 当前分支包含示例音频删除

当前分支相对主线还包含以下删除:

- `examples/emo_hate.wav`
- `examples/emo_sad.wav`
- `examples/voice_01.wav`
- `examples/voice_02.wav`
- `examples/voice_03.wav`
- `examples/voice_04.wav`
- `examples/voice_05.wav`
- `examples/voice_06.wav`
- `examples/voice_07.wav`
- `examples/voice_08.wav`
- `examples/voice_09.wav`
- `examples/voice_11.wav`
- `examples/voice_12.wav`
- `tests/sample_prompt.wav`

这些删除是在“快照当前工作区”时一并带进分支的，不是为了 speaker fusion 专门设计的删改。迁移后继续工作前，建议先确认这些文件是否应该恢复，否则 README 示例和部分手工验证流程会受影响。

### 6.5 代码内仍有局部待处理点

可见的直接待处理点包括:

- [`indextts/infer_v2.py`](../indextts/infer_v2.py) 中仍有一个情感映射 TODO
- [`openspec/changes/add-speaker-fusion-to-indextts-v2/design.md`](../openspec/changes/add-speaker-fusion-to-indextts-v2/design.md) 里的 Open Questions 尚未收敛

## 7. 迁移后建议的第一批动作

### 必做

1. 迁移本地 `checkpoints/`、`data/`、`artifacts/`
2. 在新工作区 checkout 当前分支
3. 确认是否恢复 `examples/*.wav` 和 `tests/sample_prompt.wav`
4. 用本地模型权重跑一次最小端到端 fusion smoke test

### 建议尽快做

1. 用本地 open-source slice 重跑 ten-batch 全组合实验
2. 固化一份可复现实验命令和输入切片路径
3. 产出默认推荐 scheme
4. 完成 OpenSpec `5.2` 和 `5.3`

## 8. 接手判断依据

如果迁移后只想继续代码开发，不做实验:

- 当前远端分支已经足够

如果迁移后还要继续做评分和推荐方案筛选:

- 仅靠 Git 仓库不够
- 必须补迁移本地 `data/`、`artifacts/`、`checkpoints/`

如果迁移后准备发 PR 或合并:

- 先确认示例音频删除是否应保留
- 先补一轮真实推理回归
- 先补 ten-batch 结果和默认方案结论
