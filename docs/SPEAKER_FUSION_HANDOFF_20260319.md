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

## 9. 2026-03-20 恢复记录

- 工作模式: `parallel`
- 当前目标: 跑完 `IndexTTS v2` 非训练音色融合实验，分别选出 `音色多源参考` 和 `情绪多源参考` 的推荐融合方案，并保持变量分离
- 当前假设:
  - 正式评分基于 `LibriSpeech-only` 十批次裁剪切片
  - timbre 正式清单使用 `6s` 参考裁剪版以控制 V100 显存波动
  - emotion 正式清单固定 timbre 为 A，只改变 `branch_configs.emotion`
  - 实验必须使用本机 GPU，不使用 CPU 生成音频
- 当前阻塞:
  - timbre 稳定跑尚未完成，恢复时 `artifacts/speaker_fusion_timbre_screen_trim6/run_results_stable.jsonl` 已有 `72/90`
  - `ref_mel` 路线已出现至少一个短文本不稳定样本，后续排名必须计入可靠性惩罚，不能直接推荐
- 下一步:
  - 等待或接续完成 timbre 稳定跑
  - 运行 timbre 稳定评分并选出默认/回退方案
  - 运行 emotion 正式实验与评分
  - 更新 OpenSpec `5.2`、`5.3` 和文档结论

### 9.1 2026-03-20 晚间阻塞快照

- `timbre` 稳定跑在 `77/90` 处中断:
  - 结果文件: `artifacts/speaker_fusion_timbre_screen_trim6/run_results_stable.jsonl`
  - 中断后 Python 进程变为僵尸，未继续写出新结果
- 当前 GPU 状态:
  - `nvidia-smi` 返回 `Unable to determine the device handle ... Unknown Error` / `No devices were found`
  - `.venv` 中 `torch.cuda.is_available()` 返回 `False`
  - `journalctl -k` 记录 `2026-03-20 20:45:58` 的 `NVRM: Xid 79` 与 `GPU has fallen off the bus`
  - 内核同时提示 `GPU Reset Required`
- 当前权限状态:
  - 当前用户 `nemo` 无 root 权限
  - `sudo -n true` 失败，不能在当前会话内完成驱动或 GPU reset
- 本轮在阻塞窗口内补做的代码修正:
  - `tools/score_timbre_stable.py`
    - 改为按“逻辑方案”聚合十批次结果，而不是按每条 case 独立排名
    - 保留全失败/未完成方案在报告中的可靠性惩罚，不再让缺失方案直接消失
  - `tools/score_emotion_stable.py`
    - 修正 `emotion_tensor_anchor_a` 的 sanity 判据，避免把 A 锚定方案错误罚成“必须 A/B 均衡”
    - 默认推荐只从 supported multiref 方案中选，experimental waveform 单独列为 `top_experimental`
    - 保留零输出方案在报告中的失败可见性
- 卡恢复后的直接续跑命令:

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/stable_speaker_fusion_run.py \
  --manifest artifacts/speaker_fusion_timbre_screen_trim6/manifest.jsonl \
  --results-path artifacts/speaker_fusion_timbre_screen_trim6/run_results_stable.jsonl \
  --reload-every 8 \
  --continue-on-error
```

### 9.2 2026-03-20 后续进展

- `timbre` 正式实验已补跑完成:
  - 结果文件: `artifacts/speaker_fusion_timbre_screen_trim6/run_results_stable.jsonl`
  - 总数: `90/90`
  - 稳定评分输出:
    - `artifacts/speaker_fusion_timbre_screen_trim6/scores_stable.jsonl`
    - `artifacts/speaker_fusion_timbre_screen_trim6/ranked_report_stable.csv`
    - `artifacts/speaker_fusion_timbre_screen_trim6/ranked_summary_stable.json`
- `timbre` 当前推荐结论:
  - 默认推荐: `spk_cond_emb + speech_conditioning_latent`
  - 回退推荐: `speech_conditioning_latent`
  - 不推荐进入默认集: `ref_mel` 与 `style + ref_mel`
    - 原因: 十批次中各有 `1/10` 失败，`pass_rate = 0.9`
  - 后续 CPU 单独复核:
    - `librispeech-05-timbre-006` (`ref_mel`)
    - `librispeech-05-timbre-008` (`style + ref_mel`)
    - 在 `--device cpu` 下再次稳定复现同一错误:
      - `Calculated padded input size per channel: (6). Kernel size: (7). Kernel size can't be greater than actual input size`
    - 因此 `ref_mel` 相关失败已确认为方案级不稳定，而不是 GPU / 恢复噪声
- `timbre` 稳定集主要排序摘录:
  - `spk_cond_emb + speech_conditioning_latent`
    - `pass_rate = 1.0`
    - `overall_score = 0.6866552025666145`
  - `speech_conditioning_latent`
    - `pass_rate = 1.0`
    - `overall_score = 0.6864684136020992`
  - `style`
    - `pass_rate = 1.0`
    - `overall_score = 0.681640545579843`
  - `spk_cond_emb`
    - `pass_rate = 1.0`
    - `overall_score = 0.6812977840525117`
- 评分脚本本轮已改为低显存模式:
  - `tools/score_timbre_stable.py`
  - `tools/score_emotion_stable.py`
  - 目的:
    - 不再把完整生成图都常驻在 GPU 上
    - 纠正 timbre 的“按 case 而不是按逻辑方案聚合”问题
    - 纠正 emotion 的 anchor-A 判据和 experimental 推荐污染问题
    - 支持显式 `--device cpu` / `--device cuda:0`
- `emotion` 正式实验部分完成:
  - CPU 受限续跑后已完成:
    - 限制: `<=96` 核, `<=48G` 内存
    - 离线模式: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
  - 结果文件: `artifacts/speaker_fusion_emotion_screen_trim6/run_results_stable.jsonl`
  - 总数: `50/50`
  - 最终错误数: `0`
- 第二次 GPU 硬阻塞:
  - 时间: `2026-03-20 21:24:35`
  - `journalctl -k` 记录:
    - `Xid 79`
    - `GPU has fallen off the bus`
    - `GPU Reset Required`
  - 当前直接后果:
    - `nvidia-smi` 无设备
    - `.venv` 中 `torch.cuda.is_available() == False`
    - 当前会话无法继续完成 emotion 余下 `33` 条
- GPU 恢复后的 emotion 续跑命令:
- `emotion` 稳定评分输出:
  - `artifacts/speaker_fusion_emotion_screen_trim6/scores_emotion.jsonl`
  - `artifacts/speaker_fusion_emotion_screen_trim6/ranked_report_emotion.csv`
  - `artifacts/speaker_fusion_emotion_screen_trim6/ranked_summary_emotion.json`
- `emotion` 当前推荐结论:
  - 默认推荐: `emotion_tensor_anchor_a`
  - 回退推荐: `emotion_tensor_sym`
  - experimental 观察项: `emotion_waveform_sym`
- `emotion` 排名解读:
  - 单参考基线 `emotion_a_only` 在内部情感目标空间里最高，但不是多参考方案，不进入最终多参考推荐
  - 在多参考候选里，`emotion_tensor_anchor_a` 的 `rank_score` 最高，且 `pass_rate = 1.0`
  - `emotion_tensor_sym` 次之，也 `pass_rate = 1.0`
  - `emotion_waveform_sym` 明显落后于 tensor 路线，只保留为 experimental 对照
- 当前建议的最终组合方案:
  - `音色多源参考`: `spk_cond_emb + speech_conditioning_latent`
  - `情绪多源参考`: `emotion_tensor_anchor_a`
  - 回退组合:
    - `音色`: `speech_conditioning_latent`
    - `情绪`: `emotion_tensor_sym`
- 置信度说明:
  - `timbre` 结论置信度较高，因为十批次完整跑完且可靠性惩罚已计入
  - `emotion` 结论置信度次高，因为当前使用的是固定 timbre A 的 LibriSpeech 裁剪切片和模型内部情感空间，不是专门的情感语料评测

### 9.3 CPU 续跑命令

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 OPENBLAS_NUM_THREADS=96 NUMEXPR_NUM_THREADS=96 \
VECLIB_MAXIMUM_THREADS=96 RAYON_NUM_THREADS=96 \
taskset -c 0-95 prlimit --as=51539607552 -- \
.venv/bin/python tools/stable_speaker_fusion_run.py \
  --device cpu \
  --manifest artifacts/speaker_fusion_emotion_screen_trim6/manifest.jsonl \
  --results-path artifacts/speaker_fusion_emotion_screen_trim6/run_results_stable.jsonl \
  --reload-every 4 \
  --continue-on-error
```

### 9.4 CPU 情绪评分命令

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=96 MKL_NUM_THREADS=96 OPENBLAS_NUM_THREADS=96 NUMEXPR_NUM_THREADS=96 \
VECLIB_MAXIMUM_THREADS=96 RAYON_NUM_THREADS=96 \
taskset -c 0-95 prlimit --as=51539607552 -- \
.venv/bin/python tools/score_emotion_stable.py \
  --device cpu \
  --manifest artifacts/speaker_fusion_emotion_screen_trim6/manifest.jsonl \
  --results-path artifacts/speaker_fusion_emotion_screen_trim6/run_results_stable.jsonl
```

### 9.5 GPU 恢复后的 emotion 续跑命令

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/stable_speaker_fusion_run.py \
  --manifest artifacts/speaker_fusion_emotion_screen_trim6/manifest.jsonl \
  --results-path artifacts/speaker_fusion_emotion_screen_trim6/run_results_stable.jsonl \
  --reload-every 4 \
  --continue-on-error
```
