## 1. Runtime Presets

- [x] 1.1 Add supported timbre and emotion preset builders plus argument normalization in `indextts/fusion.py`.
- [x] 1.2 Extend `IndexTTS2.infer()` and `infer_generator()` to accept rollout-facing multi-reference arguments, compose the effective recipe, and preserve backward compatibility.
- [x] 1.3 Keep raw `fusion_recipe` precedence, list-based compatibility behavior, and emitted metadata aligned with the new preset path.

## 2. User-Facing Surfaces

- [x] 2.1 Upgrade `indextts/cli.py` to run `IndexTTS2` and expose supported multi-reference timbre and emotion flags.
- [x] 2.2 Update `webui.py` to expose supported multi-reference timbre and emotion controls without surfacing raw recipe editing.
- [x] 2.3 Update `README.md` and any rollout-related docs so supported multi-reference usage is documented for Python, CLI, and WebUI.

## 3. Verification

- [x] 3.1 Add unit coverage for preset construction, runtime precedence rules, and backward-compatible single-reference behavior.
- [x] 3.2 Add one interface-level end-to-end smoke path for supported multi-reference invocation in `.venv`.
- [x] 3.3 Run the required tests, record the design/development/review iteration outcome, and update recovery docs with the final rollout status.
