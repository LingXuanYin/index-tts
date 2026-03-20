import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
if os.name == "nt":
    os.environ.setdefault("PYTORCH_JIT", "0")
import json
import re
import time
import torch
import torchaudio
import librosa
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from huggingface_hub import hf_hub_download
import safetensors
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

from indextts.fusion import (
    EXPERIMENTAL_FUSION_LEVELS,
    SUPPORTED_FUSION_LEVELS,
    FusionRecipe,
    branch_anchor_mode,
    branch_operator,
    branch_references,
    coerce_fusion_recipe,
    normalize_weights,
    recipe_cache_token,
    recipe_metadata,
)

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
            use_cuda_kernel=None,use_deepspeed=False, use_accel=False, use_torch_compile=False
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """
        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile

        self.qwen_emo = None
        self.qwen_emo_dir = os.path.join(self.model_dir, self.cfg.qwen_emo_path)

        self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)

        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}")

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d

                print(">> Preload custom CUDA kernel for BigVGAN", activation1d.anti_alias_activation_cuda)
            except Exception as e:
                print(">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                print(f"{e!r}")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print('>> semantic_codec weights restored from: {}'.format(semantic_code_ckpt))

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        
        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")
        
        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # load campplus_model
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_name)

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        # 加载术语词汇表（如果存在）
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频：
        self.reference_conditioning_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.emotion_conditioning_cache: Dict[str, torch.Tensor] = {}
        self.last_inference_metadata: Dict[str, Any] = {}

        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

    def _get_qwen_emo(self):
        if self.qwen_emo is None:
            self.qwen_emo = QwenEmotion(self.qwen_emo_dir)
        return self.qwen_emo

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def _resample_audio(self, audio: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
        if source_sr == target_sr:
            return audio
        return torchaudio.functional.resample(audio, source_sr, target_sr)

    def _prepare_audio_variants(self, audio: torch.Tensor, source_sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._resample_audio(audio, source_sr, 16000), self._resample_audio(audio, source_sr, 22050)

    def _build_reference_conditioning_from_audio(
        self,
        audio: torch.Tensor,
        source_sr: int,
        cache_key: str,
        verbose: bool = False,
        source_path: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if cache_key in self.reference_conditioning_cache:
            cached = self.reference_conditioning_cache[cache_key]
            return {key: value for key, value in cached.items()}

        audio_16k, audio_22k = self._prepare_audio_variants(audio, source_sr)
        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, s_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))
        prompt_condition = self.s2mel.models["length_regulator"](
            s_ref,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]
        speech_conditioning_latent = self.gpt.get_conditioning(
            spk_cond_emb.transpose(1, 2),
            torch.tensor([spk_cond_emb.shape[-1]], device=spk_cond_emb.device),
        )

        bundle = {
            "spk_cond_emb": spk_cond_emb,
            "speech_conditioning_latent": speech_conditioning_latent,
            "prompt_condition": prompt_condition,
            "style": style,
            "ref_mel": ref_mel,
            "cond_length": torch.tensor([spk_cond_emb.shape[-1]], device=spk_cond_emb.device),
            "source_path": source_path or cache_key,
        }
        self.reference_conditioning_cache[cache_key] = bundle
        return {key: value for key, value in bundle.items()}

    def _get_reference_conditioning(
        self,
        audio_path: str,
        verbose: bool = False,
        cache_key: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if cache_key is None:
            cache_key = audio_path
        audio, sr = self._load_and_cut_audio(audio_path, 15, verbose)
        return self._build_reference_conditioning_from_audio(
            audio,
            sr,
            cache_key=cache_key,
            verbose=verbose,
            source_path=audio_path,
        )

    def _get_emotion_conditioning(
        self,
        emo_audio_prompt: str,
        verbose: bool = False,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        if cache_key is None:
            cache_key = emo_audio_prompt
        if cache_key in self.emotion_conditioning_cache:
            return self.emotion_conditioning_cache[cache_key]

        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"].to(self.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        self.emotion_conditioning_cache[cache_key] = emo_cond_emb
        return emo_cond_emb

    def _build_emotion_conditioning_from_audio(
        self,
        audio: torch.Tensor,
        source_sr: int,
        cache_key: str,
        verbose: bool = False,
    ) -> torch.Tensor:
        if cache_key in self.emotion_conditioning_cache:
            return self.emotion_conditioning_cache[cache_key]

        emo_audio = self._resample_audio(audio, source_sr, 16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"].to(self.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        self.emotion_conditioning_cache[cache_key] = emo_cond_emb
        return emo_cond_emb

    def _resolve_emotion_conditioning(
        self,
        emo_audio_prompt: str,
        fusion_recipe: Optional[FusionRecipe],
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if fusion_recipe is None:
            emo_cond_emb = self._get_emotion_conditioning(emo_audio_prompt, verbose=verbose)
            return emo_cond_emb, {
                "references": [emo_audio_prompt],
                "weights": [1.0],
                "anchor_mode": "first",
                "operator": "identity",
                "field": "emo_cond_emb",
            }

        references = branch_references(fusion_recipe, "emotion")
        if not references:
            emo_cond_emb = self._get_emotion_conditioning(emo_audio_prompt, verbose=verbose)
            return emo_cond_emb, {
                "references": [emo_audio_prompt],
                "weights": [1.0],
                "anchor_mode": "first",
                "operator": "identity",
                "field": "emo_cond_emb",
            }

        anchor_mode = branch_anchor_mode(fusion_recipe, "emotion")
        operator = branch_operator(fusion_recipe, "emotion")
        if len(references) == 1:
            emo_cond_emb = self._get_emotion_conditioning(
                references[0].path,
                verbose=verbose,
                cache_key=recipe_cache_token(fusion_recipe, "emotion", {"path": references[0].path}),
            )
            return emo_cond_emb, {
                "references": [references[0].path],
                "weights": [1.0],
                "anchor_mode": anchor_mode,
                "operator": "identity",
                "field": "emo_cond_emb",
            }

        weights = normalize_weights(tuple(references))
        if self._field_level_enabled(fusion_recipe, "emotion", "waveform"):
            mixed_audio, mixed_sr = self._mix_reference_waveforms(references, anchor_mode, verbose=verbose)
            emo_cond_emb = self._build_emotion_conditioning_from_audio(
                mixed_audio,
                mixed_sr,
                cache_key=recipe_cache_token(fusion_recipe, "emotion", {"mixed": True, "field": "emo_cond_emb"}),
                verbose=verbose,
            )
            return emo_cond_emb, {
                "references": [reference.path for reference in references],
                "weights": weights,
                "anchor_mode": anchor_mode,
                "operator": "waveform_mix",
                "field": "emo_cond_emb",
            }

        tensors = [
            self._get_emotion_conditioning(
                reference.path,
                verbose=verbose,
                cache_key=recipe_cache_token(fusion_recipe, "emotion", {"path": reference.path, "field": "emo_cond_emb"}),
            )
            for reference in references
        ]
        merged = self._weighted_merge(tensors, weights, "time_major", anchor_mode)
        return merged, {
            "references": [reference.path for reference in references],
            "weights": weights,
            "anchor_mode": anchor_mode,
            "operator": operator,
            "field": "emo_cond_emb",
        }

    def _select_anchor_index(self, count: int, anchor_mode: str, weights: Sequence[float]) -> int:
        if count <= 1:
            return 0
        mode = (anchor_mode or "symmetric").lower()
        if mode in {"a", "first"}:
            return 0
        if mode in {"b", "second"}:
            return 1 if count > 1 else 0
        if mode == "last":
            return count - 1
        if mode == "heaviest":
            return int(max(range(count), key=lambda idx: weights[idx]))
        return 0

    def _resolve_target_length(self, lengths: Sequence[int], anchor_mode: str, weights: Sequence[float]) -> int:
        if not lengths:
            raise ValueError("Cannot resolve target length for an empty tensor list.")
        mode = (anchor_mode or "symmetric").lower()
        if mode in {"a", "first", "b", "second", "last", "heaviest"}:
            return int(lengths[self._select_anchor_index(len(lengths), anchor_mode, weights)])
        if mode == "longest":
            return int(max(lengths))
        if mode == "shortest":
            return int(min(lengths))
        weighted = sum(length * weight for length, weight in zip(lengths, weights))
        return max(1, int(round(weighted)))

    def _resize_time_major(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.size(1) == target_len:
            return tensor
        resized = F.interpolate(
            tensor.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    def _resize_channel_major(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        if tensor.size(-1) == target_len:
            return tensor
        return F.interpolate(tensor, size=target_len, mode="linear", align_corners=False)

    def _weighted_merge(
        self,
        tensors: Sequence[torch.Tensor],
        weights: Sequence[float],
        time_layout: Optional[str],
        anchor_mode: str,
    ) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        if time_layout == "time_major":
            target_len = self._resolve_target_length(
                [tensor.size(1) for tensor in tensors], anchor_mode, weights
            )
            tensors = [self._resize_time_major(tensor, target_len) for tensor in tensors]
        elif time_layout == "channel_major":
            target_len = self._resolve_target_length(
                [tensor.size(-1) for tensor in tensors], anchor_mode, weights
            )
            tensors = [self._resize_channel_major(tensor, target_len) for tensor in tensors]

        merged = torch.zeros_like(tensors[0])
        for weight, tensor in zip(weights, tensors):
            merged = merged + tensor * float(weight)
        return merged

    def _mix_reference_waveforms(
        self,
        references,
        anchor_mode: str,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        waves: List[torch.Tensor] = []
        weights = normalize_weights(tuple(references))
        for reference in references:
            audio, sr = self._load_and_cut_audio(reference.path, 15, verbose, sr=22050)
            waves.append(audio)

        target_len = self._resolve_target_length(
            [wave.size(-1) for wave in waves], anchor_mode, weights
        )
        resized = [
            self._resize_channel_major(wave.unsqueeze(0), target_len).squeeze(0)
            if wave.size(-1) != target_len
            else wave
            for wave in waves
        ]
        mixed = torch.zeros_like(resized[0])
        for weight, wave in zip(weights, resized):
            mixed = mixed + wave * float(weight)
        return mixed, 22050

    def _field_level_enabled(self, recipe: Optional[FusionRecipe], branch_name: str, level: str) -> bool:
        if recipe is None:
            return False
        branch_levels = recipe.branch(branch_name).levels
        if branch_levels:
            return level in branch_levels
        return recipe.is_enabled(level)

    def _resolve_branch_bundle(
        self,
        recipe: Optional[FusionRecipe],
        branch_name: str,
        field_name: str,
        level_name: str,
        default_path: str,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if recipe is None:
            bundle = self._get_reference_conditioning(default_path, verbose=verbose)
            return bundle[field_name], {
                "references": [default_path],
                "weights": [1.0],
                "anchor_mode": "first",
                "operator": "identity",
                "field": field_name,
            }

        references = branch_references(recipe, branch_name)
        if not references:
            bundle = self._get_reference_conditioning(default_path, verbose=verbose)
            return bundle[field_name], {
                "references": [default_path],
                "weights": [1.0],
                "anchor_mode": "first",
                "operator": "identity",
                "field": field_name,
            }

        anchor_mode = branch_anchor_mode(recipe, branch_name)
        operator = branch_operator(recipe, branch_name)
        if len(references) == 1:
            cache_key = recipe_cache_token(recipe, branch_name, {"path": references[0].path})
            bundle = self._get_reference_conditioning(references[0].path, verbose=verbose, cache_key=cache_key)
            return bundle[field_name], {
                "references": [references[0].path],
                "weights": [1.0],
                "anchor_mode": anchor_mode,
                "operator": "identity",
                "field": field_name,
            }

        use_waveform_mix = self._field_level_enabled(recipe, branch_name, "waveform")
        if use_waveform_mix:
            mixed_audio, mixed_sr = self._mix_reference_waveforms(references, anchor_mode, verbose=verbose)
            cache_key = recipe_cache_token(recipe, branch_name, {"mixed": True, "field": field_name})
            bundle = self._build_reference_conditioning_from_audio(
                mixed_audio,
                mixed_sr,
                cache_key=cache_key,
                verbose=verbose,
                source_path="waveform_mix",
            )
            return bundle[field_name], {
                "references": [reference.path for reference in references],
                "weights": normalize_weights(tuple(references)),
                "anchor_mode": anchor_mode,
                "operator": "waveform_mix",
                "field": field_name,
            }

        if not self._field_level_enabled(recipe, branch_name, level_name):
            anchor_idx = self._select_anchor_index(
                len(references),
                anchor_mode,
                normalize_weights(tuple(references)),
            )
            bundle = self._get_reference_conditioning(references[anchor_idx].path, verbose=verbose)
            return bundle[field_name], {
                "references": [references[anchor_idx].path],
                "weights": [1.0],
                "anchor_mode": anchor_mode,
                "operator": "anchor_only",
                "field": field_name,
            }

        weights = normalize_weights(tuple(references))
        bundles = [
            self._get_reference_conditioning(
                reference.path,
                verbose=verbose,
                cache_key=recipe_cache_token(recipe, branch_name, {"path": reference.path, "field": field_name}),
            )
            for reference in references
        ]
        tensors = [bundle[field_name] for bundle in bundles]
        layout = None
        if field_name in {"spk_cond_emb", "speech_conditioning_latent", "prompt_condition"}:
            layout = "time_major"
        elif field_name == "ref_mel":
            layout = "channel_major"
        merged = self._weighted_merge(tensors, weights, layout, anchor_mode)
        return merged, {
            "references": [reference.path for reference in references],
            "weights": weights,
            "anchor_mode": anchor_mode,
            "operator": operator,
            "field": field_name,
        }

    def _resolve_supported_conditioning(
        self,
        spk_audio_prompt: str,
        fusion_recipe: Optional[FusionRecipe],
        verbose: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        bundle = self._get_reference_conditioning(spk_audio_prompt, verbose=verbose)
        metadata = {"recipe": recipe_metadata(fusion_recipe), "branches": {}}
        if fusion_recipe is None:
            return bundle, metadata

        resolved = dict(bundle)
        branch_fields = {
            "speaker": ("spk_cond_emb", "spk_cond_emb"),
            "speaker_latent": ("speech_conditioning_latent", "speech_conditioning_latent"),
            "prompt": ("prompt_condition", "prompt_condition"),
            "style": ("style", "style"),
            "ref_mel": ("ref_mel", "ref_mel"),
        }
        branch_to_config = {
            "speaker": "speaker",
            "speaker_latent": "speaker",
            "prompt": "prompt",
            "style": "style",
            "ref_mel": "ref_mel",
        }
        for metadata_key, (field_name, level_name) in branch_fields.items():
            tensor, branch_meta = self._resolve_branch_bundle(
                fusion_recipe,
                branch_to_config[metadata_key],
                field_name,
                level_name,
                spk_audio_prompt,
                verbose=verbose,
            )
            resolved[field_name] = tensor
            metadata["branches"][metadata_key] = branch_meta
        resolved["cond_length"] = torch.tensor(
            [resolved["spk_cond_emb"].shape[-2]],
            device=resolved["spk_cond_emb"].device,
        )
        return resolved, metadata

    def _apply_experimental_cat_condition(
        self,
        cat_condition: torch.Tensor,
        cond: torch.Tensor,
        spk_audio_prompt: str,
        fusion_recipe: Optional[FusionRecipe],
        verbose: bool = False,
    ) -> torch.Tensor:
        if fusion_recipe is None or "cat_condition" not in fusion_recipe.experimental_levels:
            return cat_condition
        references = branch_references(fusion_recipe, "prompt")
        if len(references) < 2:
            return cat_condition
        weights = normalize_weights(tuple(references))
        cat_conditions = []
        for reference in references:
            bundle = self._get_reference_conditioning(
                reference.path,
                verbose=verbose,
                cache_key=recipe_cache_token(
                    fusion_recipe,
                    "prompt",
                    {"path": reference.path, "field": "cat_condition"},
                ),
            )
            cat_conditions.append(torch.cat([bundle["prompt_condition"], cond], dim=1))
        return self._weighted_merge(
            cat_conditions,
            weights,
            "time_major",
            branch_anchor_mode(fusion_recipe, "prompt"),
        )

    def _apply_experimental_vc_target(
        self,
        vc_target: torch.Tensor,
        cat_condition: torch.Tensor,
        fusion_recipe: Optional[FusionRecipe],
        verbose: bool,
        diffusion_steps: int,
        inference_cfg_rate: float,
    ) -> torch.Tensor:
        if fusion_recipe is None or "vc_target" not in fusion_recipe.experimental_levels:
            return vc_target
        references = branch_references(fusion_recipe, "ref_mel")
        if len(references) < 2:
            return vc_target
        weights = normalize_weights(tuple(references))
        vc_targets = []
        for reference in references:
            bundle = self._get_reference_conditioning(
                reference.path,
                verbose=verbose,
                cache_key=recipe_cache_token(
                    fusion_recipe,
                    "ref_mel",
                    {"path": reference.path, "field": "vc_target"},
                ),
            )
            target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device),
                bundle["ref_mel"],
                bundle["style"],
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            prompt_frames = min(bundle["ref_mel"].size(-1), target.size(-1))
            vc_targets.append(target[:, :, prompt_frames:])
        return self._weighted_merge(
            vc_targets,
            weights,
            "channel_major",
            branch_anchor_mode(fusion_recipe, "ref_mel"),
        )

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Silences to be insert between generated segments.
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        return torch.zeros(channel_size, sil_dur)

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between generated segments.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)

    def _load_and_cut_audio(self,audio_path,max_audio_length_seconds,verbose=False,sr=None):
        if not sr:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path,sr=sr)
        audio = torch.tensor(audio).unsqueeze(0)
        max_audio_samples = int(max_audio_length_seconds * sr)

        if audio.shape[1] > max_audio_samples:
            if verbose:
                print(f"Audio too long ({audio.shape[1]} samples), truncating to {max_audio_samples} samples")
            audio = audio[:, :max_audio_samples]
        return audio, sr
    
    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        # apply biased emotion factors for better user experience,
        # by de-emphasizing emotions that can cause strange results
        if apply_bias:
            # [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

        # the total emotion sum must be 0.8 or less
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]

        return emo_vector

    def _write_metadata_file(self, metadata_output_path: Optional[str], metadata: Dict[str, Any]) -> None:
        if not metadata_output_path:
            return
        output_dir = os.path.dirname(metadata_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(metadata_output_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    # 原始推理模式
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
              fusion_recipe=None, return_metadata=False, metadata_output_path=None, **generation_kwargs):
        if stream_return:
            return self.infer_generator(
                spk_audio_prompt, text, output_path,
                emo_audio_prompt, emo_alpha,
                emo_vector,
                use_emo_text, emo_text, use_random, interval_silence,
                verbose, max_text_tokens_per_segment, stream_return, more_segment_before,
                fusion_recipe=fusion_recipe, return_metadata=return_metadata,
                metadata_output_path=metadata_output_path, **generation_kwargs
            )
        else:
            try:
                return list(self.infer_generator(
                    spk_audio_prompt, text, output_path,
                    emo_audio_prompt, emo_alpha,
                    emo_vector,
                    use_emo_text, emo_text, use_random, interval_silence,
                    verbose, max_text_tokens_per_segment, stream_return, more_segment_before,
                    fusion_recipe=fusion_recipe, return_metadata=return_metadata,
                    metadata_output_path=metadata_output_path, **generation_kwargs
                ))[0]
            except IndexError:
                return None

    def infer_generator(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_segment=120, stream_return=False, quick_streaming_tokens=0,
              fusion_recipe=None, return_metadata=False, metadata_output_path=None, **generation_kwargs):
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                  f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                  f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                  f"emo_text:{emo_text}")
        start_time = time.perf_counter()
        if fusion_recipe is None and isinstance(spk_audio_prompt, (list, tuple)):
            fusion_recipe = {
                "references": list(spk_audio_prompt),
                "enabled_levels": list(SUPPORTED_FUSION_LEVELS),
            }
            spk_audio_prompt = spk_audio_prompt[0]
        fusion_recipe = coerce_fusion_recipe(fusion_recipe)

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self._get_qwen_emo().inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio,sr = self._load_and_cut_audio(spk_audio_prompt,15,verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.device)
            attention_mask = attention_mask.to(self.device)
            spk_cond_emb = self.get_emb(input_features, attention_mask)

            _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
            ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                     num_mel_bins=80,
                                                     dither=0,
                                                     sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                     ylens=ref_target_lengths,
                                                                     n_quantizers=3,
                                                                     f0=None)[0]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        fusion_metadata = {"recipe": recipe_metadata(fusion_recipe), "branches": {}}
        speech_conditioning_latent = None
        if fusion_recipe is not None:
            conditioning_bundle, fusion_metadata = self._resolve_supported_conditioning(
                spk_audio_prompt,
                fusion_recipe,
                verbose=verbose,
            )
            spk_cond_emb = conditioning_bundle["spk_cond_emb"]
            speech_conditioning_latent = conditioning_bundle["speech_conditioning_latent"]
            prompt_condition = conditioning_bundle["prompt_condition"]
            style = conditioning_bundle["style"]
            ref_mel = conditioning_bundle["ref_mel"]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]

            emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
            emo_matrix = torch.cat(emo_matrix, 0)
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix
            emovec_mat = torch.sum(emovec_mat, 0)
            emovec_mat = emovec_mat.unsqueeze(0)

        emotion_metadata = {
            "references": [emo_audio_prompt],
            "weights": [1.0],
            "anchor_mode": "first",
            "operator": "identity",
            "field": "emo_cond_emb",
        }
        if fusion_recipe is not None:
            emo_cond_emb, emotion_metadata = self._resolve_emotion_conditioning(
                emo_audio_prompt,
                fusion_recipe,
                verbose=verbose,
            )
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
            fusion_metadata["branches"]["emotion"] = emotion_metadata
        elif self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_cond_emb = self._get_emotion_conditioning(emo_audio_prompt, verbose=verbose)
            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment, quick_streaming_tokens = quick_streaming_tokens)
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):")
            print( "     Tokens which can't be encoded: ", [t for t, id in zip(text_tokens_list, text_token_ids) if id == self.tokenizer.unk_token_id])
            print(f"     Consider updating the BPE model or modifying the text to avoid unknown tokens.")
                  
        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050
        diffusion_steps = fusion_recipe.diffusion_steps if fusion_recipe is not None else generation_kwargs.pop("diffusion_steps", 25)
        inference_cfg_rate = (
            fusion_recipe.inference_cfg_rate
            if fusion_recipe is not None
            else generation_kwargs.pop("inference_cfg_rate", 0.7)
        )

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None # for stream_return
        run_metadata = {
            "text": text,
            "speaker_prompt": spk_audio_prompt,
            "emotion_prompt": emo_audio_prompt,
            "emotion_alpha": emo_alpha,
            "emotion_vector": emo_vector,
            "fusion": fusion_metadata,
            "experimental_levels": list(fusion_recipe.experimental_levels) if fusion_recipe is not None else [],
            "segments": [],
            "diffusion_steps": diffusion_steps,
            "inference_cfg_rate": inference_cfg_rate,
        }
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(0.2 + 0.7 * seg_idx / segments_count,
                                  f"speech synthesis {seg_idx + 1}/{segments_count}...")

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}")
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(text_tokens[0].tolist())
                print("text_token_syms is same as segment tokens", text_token_syms == sent)

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        alpha=emo_alpha
                    )

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        speech_conditioning_latent_override=speech_conditioning_latent,
                        **generation_kwargs
                    )

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning
                    )
                    has_warned = True

                code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                with torch.amp.autocast(text_tokens.device.type, enabled=self.dtype is not None, dtype=self.dtype):
                    latent = self.gpt(
                        speech_conditioning_latent,
                        text_tokens,
                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                        codes,
                        torch.tensor([codes.shape[-1]], device=text_tokens.device),
                        emo_cond_emb,
                        cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens.device),
                        emo_vec=emovec,
                        use_speed=use_speed,
                    )
                    gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(text_tokens.device.type, enabled=dtype is not None, dtype=dtype):
                    m_start_time = time.perf_counter()
                    latent = self.s2mel.models['gpt_layer'](latent)
                    S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                    S_infer = S_infer.transpose(1, 2)
                    S_infer = S_infer + latent
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models['length_regulator'](S_infer,
                                                                 ylens=target_lengths,
                                                                 n_quantizers=3,
                                                                 f0=None)[0]
                    cat_condition = torch.cat([prompt_condition, cond], dim=1)
                    cat_condition = self._apply_experimental_cat_condition(
                        cat_condition,
                        cond,
                        spk_audio_prompt,
                        fusion_recipe,
                        verbose=verbose,
                    )
                    vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                   torch.LongTensor([cat_condition.size(1)]).to(
                                                                       cond.device),
                                                                   ref_mel, style, None, diffusion_steps,
                                                                    inference_cfg_rate=inference_cfg_rate)
                    prompt_frames = min(ref_mel.size(-1), vc_target.size(-1))
                    vc_target = vc_target[:, :, prompt_frames:]
                    vc_target = self._apply_experimental_vc_target(
                        vc_target,
                        cat_condition,
                        fusion_recipe,
                        verbose,
                        diffusion_steps,
                        inference_cfg_rate,
                    )
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
                run_metadata["segments"].append(
                    {
                        "index": seg_idx,
                        "text": self.tokenizer.decode(text_tokens[0].tolist()),
                        "code_length": int(code_lens[0].item()),
                        "wav_samples": int(wav.shape[-1]),
                        "cat_condition_length": int(cat_condition.size(1)),
                    }
                )
                if stream_return:
                    yield wav.cpu()
                    if silence == None:
                        silence = self.interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")
        run_metadata["timings"] = {
            "gpt_gen_time": gpt_gen_time,
            "gpt_forward_time": gpt_forward_time,
            "s2mel_time": s2mel_time,
            "bigvgan_time": bigvgan_time,
            "total_time": end_time - start_time,
            "wav_length_seconds": wav_length,
            "rtf": (end_time - start_time) / wav_length if wav_length > 0 else None,
        }

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            run_metadata["output_path"] = output_path
            self.last_inference_metadata = run_metadata
            self._write_metadata_file(metadata_output_path, run_metadata)
            if stream_return:
                return None
            if return_metadata:
                yield {"output_path": output_path, "metadata": run_metadata}
            else:
                yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            self.last_inference_metadata = run_metadata
            self._write_metadata_file(metadata_output_path, run_metadata)
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            if return_metadata:
                yield {"audio": (sampling_rate, wav_data), "metadata": run_metadata}
            else:
                yield (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir):
        from modelscope import AutoModelForCausalLM

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            print(">> no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)


if __name__ == "__main__":
    prompt_wav = "examples/voice_01.wav"
    text = '欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。'
    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml", 
        model_dir="checkpoints", 
        use_cuda_kernel=False,
        use_torch_compile=True
    )
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
    char_size = 5
    import string
    time_buckets = []
    for i in range(10):
        text = ''.join(random.choices(string.ascii_letters, k=char_size))
        start_time = time.time()
        tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
        time_buckets.append(time.time() - start_time)
    print(time_buckets)
