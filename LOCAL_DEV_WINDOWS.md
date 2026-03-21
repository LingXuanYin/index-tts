# Windows Local Development

Last updated: 2026-03-21

This repository is developed without Docker. Use the project virtual environment at the repository root.

## Environment

1. Install Python 3.10 or newer.
2. Install `uv`.
3. Create or sync the local environment from the repository root:

```powershell
uv sync --extra webui
```

If you need every optional feature:

```powershell
uv sync --all-extras
```

The project environment lives at:

- `.venv\`
- Python executable: `.venv\Scripts\python.exe`

## Common Commands

Run the WebUI:

```powershell
uv run webui.py
```

Run the CLI:

```powershell
uv run indextts "hello world" --voice path\to\voice.wav --output_path outputs\hello.wav
```

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_fusion_rollout.py tests/test_cli_v2.py tests/test_speaker_fusion_experiment.py
```

## Recovery

If the session is interrupted, reload context from:

1. `docs/TEAM_MULTIREF_FUSION_ROLLOUT.md`
2. `docs/WORKSTATE_MULTIREF_FUSION_20260321.md`
3. `openspec/changes/rollout-multiref-fusion-indextts-v2/`
4. `docs/SPEAKER_FUSION_HANDOFF_20260319.md`
