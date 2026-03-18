## 1. Fusion Tensor Extraction

- [x] 1.1 Refactor `indextts/infer_v2.py` to extract speaker-related tensors (`spk_cond_emb`, `speech_conditioning_latent`, `prompt_condition`, `style`, `ref_mel`) through shared helper functions.
- [x] 1.2 Add fusion-recipe data structures that describe references, weights, branch roles, anchor mode, and supported versus experimental levels.
- [x] 1.3 Extend cache identity logic so speaker-related caches include fusion-relevant fields and remain safe across experiment runs.

## 2. Supported-Tier Speaker Fusion

- [x] 2.1 Add supported-tier tensor fusion operators for `spk_cond_emb`, `speech_conditioning_latent`, `prompt_condition`, `style`, and `ref_mel`.
- [x] 2.2 Extend `IndexTTS2` inference entrypoints to accept multi-reference fusion requests while preserving the existing single-speaker path.
- [x] 2.3 Add branch-level reference-role handling so speaker, prompt, style, and mel branches can use independent fusion settings in one request.
- [x] 2.4 Add metadata emission for each synthesis run so outputs record the exact supported-tier fusion recipe that produced them.

## 3. Experimental-Tier Fusion Coverage

- [x] 3.1 Add opt-in experimental operators for waveform mixing, `cat_condition`, and `vc_target` without enabling them in the default inference path.
- [x] 3.2 Mark experimental runs in metadata and isolate them from the default recommendation workflow.

## 4. Experiment Matrix And Evaluation

- [x] 4.1 Create a manifest format for speaker fusion experiments that records text, references, branch selections, weights, anchor modes, and output paths.
- [x] 4.2 Add open-source dataset sampling logic and define the first ten-batch speaker-fusion evaluation slice with dataset provenance metadata.
- [x] 4.3 Implement full-combination manifest generation across configured fusion levels, weights, anchor modes, and branch-role assignments.
- [x] 4.4 Add automatic scoring that reports speaker-A similarity, speaker-B similarity, fused-target similarity, semantic stability, runtime, and generation health metrics.
- [x] 4.5 Add ranked reporting that summarizes full-combination fusion schemes, recommends a default scheme, and exports a listening-review subset from the ten-batch run.

## 5. Verification And Rollout

- [x] 5.1 Add focused tests for backward-compatible single-speaker inference, supported-tier fusion execution, cache-key isolation, and experiment manifest reproducibility.
- [ ] 5.2 Run the first ten-batch open-source full-combination experiment to verify output generation, score collection, and report ranking.
- [ ] 5.3 Document the supported and experimental fusion levels, the ten-batch open-source experiment workflow, and the selected default recommendation for future apply work.
