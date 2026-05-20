from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class GenerationResult:
    generated_text: str
    input_tokens: int
    new_tokens: int
    elapsed_sec: float
    peak_mem_gib: float | None = None


@dataclass
class ModelLoadReport:
    model_id: str
    revision: str | None = None
    loaded: bool = False
    manual_config_patch: bool = False
    manual_weight_conversion: bool = False
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    num_parameters: int | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRunner(Protocol):
    label: str
    load_report: ModelLoadReport

    def load(self) -> ModelLoadReport: ...

    def generate(self, prompt: str) -> GenerationResult: ...


class MockRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.label = config.get("label", "Mock")
        self.load_report = ModelLoadReport(model_id=config.get("model_id", "mock"), loaded=False)

    def load(self) -> ModelLoadReport:
        self.load_report.loaded = True
        self.load_report.notes.append("Mock runner returns the target answer when it is embedded in the prompt.")
        return self.load_report

    def generate(self, prompt: str) -> GenerationResult:
        import re

        start = time.time()
        query_match = re.search(r"End of document\..*PASSKEY_RECORD\[([^\]]+)\]\s*=", prompt, flags=re.DOTALL)
        if query_match:
            key = re.escape(query_match.group(1))
            match = re.search(rf"PASSKEY_RECORD\[{key}\]\s*=\s*(\d+)", prompt)
        else:
            match = re.search(r"PASSKEY_RECORD\[[^\]]+\]\s*=\s*(\d+)", prompt)
        generated = f" {match.group(1)}" if match else ""
        return GenerationResult(
            generated_text=generated,
            input_tokens=len(prompt.split()),
            new_tokens=len(generated.split()),
            elapsed_sec=time.time() - start,
            peak_mem_gib=0.0,
        )


class TransformersCausalLMRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = config["model_id"]
        self.tokenizer_id = config.get("tokenizer_id", self.model_id)
        self.revision = config.get("revision")
        self.tokenizer_revision = config.get("tokenizer_revision", self.revision)
        self.label = config.get("label", self.model_id)
        self.trust_remote_code = bool(config.get("trust_remote_code", False))
        self.dtype_name = config.get("dtype", "bfloat16")
        self.device = config.get("device", "cuda")
        self.max_new_tokens = int(config.get("max_new_tokens", 8))
        self.do_sample = bool(config.get("do_sample", False))
        self.use_cache = bool(config.get("use_cache", True))
        self.config_patch = dict(config.get("config_patch", {}) or {})
        self.manual_config_patch = bool(self.config_patch)
        self.manual_weight_conversion = bool(config.get("manual_weight_conversion", False))
        self.load_report = ModelLoadReport(
            model_id=self.model_id,
            revision=self.revision,
            manual_config_patch=self.manual_config_patch,
            manual_weight_conversion=self.manual_weight_conversion,
            notes=[],
        )
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> ModelLoadReport:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            dtype = getattr(torch, self.dtype_name)
            tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            if self.tokenizer_revision:
                tokenizer_kwargs["revision"] = self.tokenizer_revision
            if self.revision:
                model_kwargs["revision"] = self.revision

            if self.tokenizer_id != self.model_id:
                self.load_report.notes.append(f"Using tokenizer_id={self.tokenizer_id!r}")
            if self.tokenizer_revision:
                self.load_report.notes.append(f"Using tokenizer_revision={self.tokenizer_revision!r}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **tokenizer_kwargs)
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code, revision=self.revision)
            for key, value in self.config_patch.items():
                setattr(config, key, value)
                self.load_report.notes.append(f"Applied explicit config patch: {key}={value!r}")

            if self.manual_weight_conversion:
                raise NotImplementedError(
                    "manual_weight_conversion is intentionally not implemented in the generic runner. "
                    "Add a dedicated, audited loader before using converted weights."
                )

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, config=config, **model_kwargs).to(self.device)
            self.model.eval()
            self.load_report.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
            self.load_report.loaded = True
        except Exception as exc:
            self.load_report.error = repr(exc)
            raise
        return self.load_report

    def generate(self, prompt: str) -> GenerationResult:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Model must be loaded before generation.")

        torch = self.torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        start = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=self.use_cache,
                return_dict_in_generate=False,
                output_scores=False,
                output_hidden_states=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        generated_ids = output_ids[0, input_len:].detach().cpu()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        peak_mem_gib = None
        if torch.cuda.is_available():
            peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

        del inputs
        del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerationResult(
            generated_text=generated_text,
            input_tokens=input_len,
            new_tokens=int(len(generated_ids)),
            elapsed_sec=elapsed,
            peak_mem_gib=peak_mem_gib,
        )


class MambaSSMLMRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = config["model_id"]
        self.tokenizer_id = config.get("tokenizer_id", "EleutherAI/gpt-neox-20b")
        self.revision = config.get("revision")
        self.tokenizer_revision = config.get("tokenizer_revision", self.revision)
        self.label = config.get("label", self.model_id)
        self.dtype_name = config.get("dtype", "bfloat16")
        self.device = config.get("device", "cuda")
        self.max_new_tokens = int(config.get("max_new_tokens", 8))
        self.top_k = int(config.get("top_k", 1))
        self.top_p = float(config.get("top_p", 0.0))
        self.min_p = float(config.get("min_p", 0.0))
        self.temperature = float(config.get("temperature", 1.0))
        self.load_report = ModelLoadReport(model_id=self.model_id, revision=self.revision, notes=[])
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> ModelLoadReport:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            from transformers import AutoTokenizer

            self.torch = torch
            dtype = getattr(torch, self.dtype_name)
            tokenizer_kwargs = {}
            if self.tokenizer_revision:
                tokenizer_kwargs["revision"] = self.tokenizer_revision
            model_source = self.model_id
            if self.revision:
                model_source = snapshot_download(
                    self.model_id,
                    revision=self.revision,
                    allow_patterns=["config.json", "pytorch_model.bin"],
                )
                self.load_report.notes.append(f"Resolved model revision to snapshot: {model_source}")

            self.load_report.notes.append("Loaded through mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel.")
            if self.tokenizer_id != self.model_id:
                self.load_report.notes.append(f"Using tokenizer_id={self.tokenizer_id!r}")
            if self.tokenizer_revision:
                self.load_report.notes.append(f"Using tokenizer_revision={self.tokenizer_revision!r}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **tokenizer_kwargs)
            self.model = MambaLMHeadModel.from_pretrained(model_source, device=self.device, dtype=dtype)
            self.model.eval()
            self.load_report.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
            self.load_report.loaded = True
        except Exception as exc:
            self.load_report.error = repr(exc)
            raise
        return self.load_report

    def generate(self, prompt: str) -> GenerationResult:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Model must be loaded before generation.")

        torch = self.torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        input_len = int(input_ids.shape[1])
        start = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_len + self.max_new_tokens,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                temperature=self.temperature,
                return_dict_in_generate=False,
                output_scores=False,
            )

        elapsed = time.time() - start
        generated_ids = output_ids[0, input_len:].detach().cpu()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        peak_mem_gib = None
        if torch.cuda.is_available():
            peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

        del input_ids
        del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerationResult(
            generated_text=generated_text,
            input_tokens=input_len,
            new_tokens=int(len(generated_ids)),
            elapsed_sec=elapsed,
            peak_mem_gib=peak_mem_gib,
        )


class GatedDeltaNetConvertedRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = config["model_id"]
        self.tokenizer_id = config.get("tokenizer_id", self.model_id)
        self.revision = config.get("revision")
        self.tokenizer_revision = config.get("tokenizer_revision", self.revision)
        self.label = config.get("label", self.model_id)
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.dtype_name = config.get("dtype", "bfloat16")
        self.device = config.get("device", "cuda")
        self.max_new_tokens = int(config.get("max_new_tokens", 8))
        self.do_sample = bool(config.get("do_sample", False))
        self.use_cache = bool(config.get("use_cache", True))
        self.checkpoint_filename = config.get("checkpoint_filename", "model.safetensors")
        self.patch_swiglu_triton = bool(config.get("patch_swiglu_triton", False))
        self.config_patch = dict(config.get("config_patch", {}) or {})
        self.load_report = ModelLoadReport(
            model_id=self.model_id,
            revision=self.revision,
            manual_config_patch=True,
            manual_weight_conversion=True,
            notes=[],
        )
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> ModelLoadReport:
        try:
            import torch
            import fla.models  # noqa: F401
            import fla.modules.mlp as fla_mlp
            from huggingface_hub import hf_hub_download, snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            if self.patch_swiglu_triton:
                fla_mlp.swiglu = lambda gate, up: torch.nn.functional.silu(gate) * up
                self.load_report.notes.append("Patched FLA SwiGLU activation to a pure PyTorch inference fallback.")
            dtype = getattr(torch, self.dtype_name)
            tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
            config_kwargs = {"trust_remote_code": self.trust_remote_code}
            download_kwargs = {}
            if self.tokenizer_revision:
                tokenizer_kwargs["revision"] = self.tokenizer_revision
            if self.revision:
                config_kwargs["revision"] = self.revision
                download_kwargs["revision"] = self.revision

            if self.tokenizer_id != self.model_id:
                self.load_report.notes.append(f"Using tokenizer_id={self.tokenizer_id!r}")
            if self.tokenizer_revision:
                self.load_report.notes.append(f"Using tokenizer_revision={self.tokenizer_revision!r}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **tokenizer_kwargs)
            config = AutoConfig.from_pretrained(self.model_id, **config_kwargs)
            self._patch_intermediate_size(config)
            for key, value in self.config_patch.items():
                setattr(config, key, value)
                self.load_report.notes.append(f"Applied explicit config patch: {key}={value!r}")

            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=self.trust_remote_code)
            converted_state, converted_count = self._load_converted_state_dict(
                hf_hub_download=hf_hub_download,
                snapshot_download=snapshot_download,
                load_file=load_file,
                intermediate_size=int(config.intermediate_size),
                download_kwargs=download_kwargs,
            )

            incompatible = self.model.load_state_dict(converted_state, strict=False)
            del converted_state
            self.load_report.missing_keys = list(incompatible.missing_keys)
            self.load_report.unexpected_keys = list(incompatible.unexpected_keys)
            self.load_report.notes.append(f"Split {converted_count} fused MLP gate/up projection tensors.")
            self._validate_load_keys()

            self.model.tie_weights()
            self.model = self.model.to(device=self.device, dtype=dtype)
            self.model.eval()
            self.load_report.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
            self.load_report.loaded = True
        except Exception as exc:
            self.load_report.error = repr(exc)
            raise
        return self.load_report

    def _patch_intermediate_size(self, config: Any) -> None:
        if getattr(config, "intermediate_size", None) is not None:
            self.load_report.notes.append(f"Config already has intermediate_size={config.intermediate_size!r}.")
            return
        raw_intermediate = int(config.hidden_size * config.hidden_ratio * 2 / 3)
        config.intermediate_size = 256 * math.ceil(raw_intermediate / 256)
        self.load_report.notes.append(
            "Patched intermediate_size from hidden_size and hidden_ratio: "
            f"{config.intermediate_size}."
        )

    def _load_converted_state_dict(
        self,
        *,
        hf_hub_download: Any,
        snapshot_download: Any,
        load_file: Any,
        intermediate_size: int,
        download_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        converted_state: dict[str, Any] = {}
        converted_count = 0
        if self.checkpoint_filename:
            checkpoint_paths = [hf_hub_download(self.model_id, self.checkpoint_filename, **download_kwargs)]
        else:
            snapshot_path = snapshot_download(self.model_id, allow_patterns=["*.safetensors"], **download_kwargs)
            checkpoint_paths = sorted(str(path) for path in Path(snapshot_path).glob("*.safetensors"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"No safetensors checkpoints found in snapshot for {self.model_id!r}.")
            self.load_report.notes.append(f"Loaded sharded safetensors checkpoint with {len(checkpoint_paths)} files.")

        for checkpoint_path in checkpoint_paths:
            raw_state = load_file(checkpoint_path, device="cpu")
            shard_state, shard_count = self._convert_state_dict(raw_state, intermediate_size)
            converted_state.update(shard_state)
            converted_count += shard_count
            del raw_state
        return converted_state, converted_count

    def _convert_state_dict(self, raw_state: dict[str, Any], intermediate_size: int) -> tuple[dict[str, Any], int]:
        converted_state = {}
        converted_count = 0
        fused_rows = 2 * intermediate_size
        for key, value in raw_state.items():
            if key.endswith("mlp.gate_proj.weight") and value.shape[0] == fused_rows:
                gate_weight, up_weight = value.chunk(2, dim=0)
                converted_state[key] = gate_weight
                converted_state[key.replace("gate_proj.weight", "up_proj.weight")] = up_weight
                converted_count += 1
            else:
                converted_state[key] = value
        return converted_state, converted_count

    def _validate_load_keys(self) -> None:
        allowed_unexpected = [key for key in self.load_report.unexpected_keys if key.endswith(".attn.D")]
        invalid_unexpected = sorted(set(self.load_report.unexpected_keys) - set(allowed_unexpected))
        if allowed_unexpected:
            self.load_report.notes.append(f"Ignored {len(allowed_unexpected)} unexpected attn.D tensors.")
        if self.load_report.missing_keys or invalid_unexpected:
            raise ValueError(
                "Converted Gated DeltaNet checkpoint did not load cleanly: "
                f"missing={self.load_report.missing_keys}, unexpected={invalid_unexpected}"
            )

    def generate(self, prompt: str) -> GenerationResult:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Model must be loaded before generation.")

        torch = self.torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_len = int(inputs["input_ids"].shape[1])
        start = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=self.use_cache,
                return_dict_in_generate=False,
                output_scores=False,
                output_hidden_states=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        generated_ids = output_ids[0, input_len:].detach().cpu()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        peak_mem_gib = None
        if torch.cuda.is_available():
            peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

        del inputs
        del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerationResult(
            generated_text=generated_text,
            input_tokens=input_len,
            new_tokens=int(len(generated_ids)),
            elapsed_sec=elapsed,
            peak_mem_gib=peak_mem_gib,
        )


class HazyResearchLMRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = config["model_id"]
        self.tokenizer_id = config.get("tokenizer_id", "gpt2")
        self.revision = config.get("revision")
        self.tokenizer_revision = config.get("tokenizer_revision")
        self.architecture = config.get("architecture", "based")
        self.label = config.get("label", self.model_id)
        self.dtype_name = config.get("dtype", "bfloat16")
        self.device = config.get("device", "cuda")
        self.max_new_tokens = int(config.get("max_new_tokens", 8))
        self.load_report = ModelLoadReport(model_id=self.model_id, revision=self.revision, notes=[])
        self.tokenizer = None
        self.model = None
        self.torch = None

    def load(self) -> ModelLoadReport:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer

            self.torch = torch
            dtype = getattr(torch, self.dtype_name)
            tokenizer_kwargs = {}
            if self.tokenizer_revision:
                tokenizer_kwargs["revision"] = self.tokenizer_revision

            model_source = self.model_id
            if self.revision:
                model_source = snapshot_download(self.model_id, revision=self.revision)
                self.load_report.notes.append(f"Resolved model revision to snapshot: {model_source}")

            if self.tokenizer_id != self.model_id:
                self.load_report.notes.append(f"Using tokenizer_id={self.tokenizer_id!r}")
            if self.tokenizer_revision:
                self.load_report.notes.append(f"Using tokenizer_revision={self.tokenizer_revision!r}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **tokenizer_kwargs)

            model_cls = self._model_class()
            self.model = model_cls.from_pretrained_hf(model_source).to(device=self.device, dtype=dtype)
            self.model.eval()
            self.load_report.notes.append(f"Loaded HazyResearch {self.architecture!r} checkpoint.")
            self.load_report.num_parameters = sum(parameter.numel() for parameter in self.model.parameters())
            self.load_report.loaded = True
        except Exception as exc:
            self.load_report.error = repr(exc)
            raise
        return self.load_report

    def _model_class(self) -> Any:
        if self.architecture == "based":
            from based.models.gpt import GPTLMHeadModel

            return GPTLMHeadModel
        if self.architecture == "attention":
            from based.models.transformer.gpt import GPTLMHeadModel

            return GPTLMHeadModel
        if self.architecture == "mamba":
            from based.models.mamba import MambaLMHeadModel

            return MambaLMHeadModel
        raise ValueError(f"Unknown HazyResearch architecture: {self.architecture!r}")

    def generate(self, prompt: str) -> GenerationResult:
        if self.model is None or self.tokenizer is None or self.torch is None:
            raise RuntimeError("Model must be loaded before generation.")

        torch = self.torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_len = int(input_ids.shape[1])
        start = time.time()

        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, max_length=input_len + self.max_new_tokens)

        elapsed = time.time() - start
        generated_ids = output_ids[0, input_len:].detach().cpu()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        peak_mem_gib = None
        if torch.cuda.is_available():
            peak_mem_gib = torch.cuda.max_memory_allocated() / 1024**3

        del input_ids
        del output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerationResult(
            generated_text=generated_text,
            input_tokens=input_len,
            new_tokens=int(len(generated_ids)),
            elapsed_sec=elapsed,
            peak_mem_gib=peak_mem_gib,
        )


def build_runner(config: dict[str, Any]) -> ModelRunner:
    runner_type = config.get("runner", "transformers_causal_lm")
    if runner_type == "mock":
        return MockRunner(config)
    if runner_type == "transformers_causal_lm":
        return TransformersCausalLMRunner(config)
    if runner_type == "mamba_ssm_lm":
        return MambaSSMLMRunner(config)
    if runner_type == "gated_deltanet_converted":
        return GatedDeltaNetConvertedRunner(config)
    if runner_type == "hazyresearch_lm":
        return HazyResearchLMRunner(config)
    raise ValueError(f"Unknown runner type: {runner_type}")
