from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
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
        self.decode_strategy = config.get("decode_strategy")
        if self.decode_strategy is None:
            self.decode_strategy = "recompute" if self.architecture == "based" else "cached"
        if self.decode_strategy not in {"cached", "recompute"}:
            raise ValueError(f"Unsupported HazyResearch decode_strategy: {self.decode_strategy!r}")
        self.config_patch = dict(config.get("config_patch", {}) or {})
        self.constructor_kwargs = dict(config.get("constructor_kwargs", {}) or {})
        self.load_report = ModelLoadReport(
            model_id=self.model_id,
            revision=self.revision,
            manual_config_patch=bool(self.config_patch),
            notes=[],
        )
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
            self._patch_hazy_optional_types()
            self._patch_hazy_config_loader()
            if self.architecture == "attention" and self.constructor_kwargs:
                self.model = self._load_attention_model(model_source=model_source).to(device=self.device, dtype=dtype)
            else:
                self.model = model_cls.from_pretrained_hf(model_source).to(device=self.device, dtype=dtype)
            self.model.eval()
            self.load_report.notes.append(f"Loaded HazyResearch {self.architecture!r} checkpoint.")
            self.load_report.notes.append(f"Using HazyResearch decode_strategy={self.decode_strategy!r}.")
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

    def _patch_hazy_config_loader(self) -> None:
        if not self.config_patch:
            return

        import based.utils.hf as based_hf

        original_load_config_hf = based_hf.load_config_hf

        def load_config_hf_with_patch(*args: Any, **kwargs: Any) -> dict[str, Any]:
            config_data = dict(original_load_config_hf(*args, **kwargs))
            config_data.update(self.config_patch)
            return config_data

        based_hf.load_config_hf = load_config_hf_with_patch
        if self.architecture == "based":
            import based.models.gpt as based_gpt

            based_gpt.load_config_hf = load_config_hf_with_patch
        for key, value in self.config_patch.items():
            self.load_report.notes.append(f"Applied explicit HazyResearch config patch: {key}={value!r}")

    def _patch_hazy_optional_types(self) -> None:
        class UnavailableColumnParallelLinear:
            pass

        if self.architecture == "based":
            import based.models.gpt as based_gpt

            if based_gpt.ColumnParallelLinear is None:
                based_gpt.ColumnParallelLinear = UnavailableColumnParallelLinear
                self.load_report.notes.append("Patched missing ColumnParallelLinear optional type.")
        if self.architecture == "attention":
            import based.models.transformer.gpt as transformer_gpt

            if transformer_gpt.ColumnParallelLinear is None:
                transformer_gpt.ColumnParallelLinear = UnavailableColumnParallelLinear
                self.load_report.notes.append("Patched missing ColumnParallelLinear optional type.")

    def _load_attention_model(self, *, model_source: str) -> Any:
        import re
        import torch
        import based.utils.hf as based_hf
        import based.models.transformer.gpt as transformer_gpt

        config_data = based_hf.load_config_hf(model_source)
        config = transformer_gpt.GPT2Config(**config_data)
        model = transformer_gpt.GPTLMHeadModel(
            config=config,
            device=self.device,
            dtype=torch.float16,
            **self.constructor_kwargs,
        )
        state_dict = transformer_gpt.state_dict_from_pretrained(model_source, dtype=torch.float16)
        state_dict = {re.sub("^model\\.", "", key): value for key, value in state_dict.items()}
        state_dict = {key: value for key, value in state_dict.items() if "metrics" not in key}
        incompatible = model.load_state_dict(state_dict, strict=True)
        self.load_report.missing_keys = list(incompatible.missing_keys)
        self.load_report.unexpected_keys = list(incompatible.unexpected_keys)
        self.load_report.notes.append(f"Used attention constructor kwargs: {self.constructor_kwargs!r}")
        return model

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

        if self.decode_strategy == "cached":
            with torch.inference_mode():
                output_ids = self.model.generate(input_ids, max_length=input_len + self.max_new_tokens)
        elif self.decode_strategy == "recompute":
            output_ids = self._generate_recompute(input_ids)
        else:  # pragma: no cover - validated during initialization.
            raise ValueError(f"Unsupported HazyResearch decode_strategy: {self.decode_strategy!r}")

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

    def _generate_recompute(self, input_ids: Any) -> Any:
        """Greedy decode by recomputing the full sequence at each new token.

        The public BASED fallback generation path can produce unstable cached
        recurrent decoding for some sequence lengths. Recompute decoding is
        slower, but keeps accuracy runs on the same full-forward path used for
        prompt prefill and avoids cache-specific artifacts.
        """

        torch = self.torch
        assert torch is not None
        output_ids = input_ids
        for _ in range(self.max_new_tokens):
            with torch.inference_mode():
                logits = self.model(output_ids, num_last_tokens=1).logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token], dim=1)
        return output_ids


def build_runner(config: dict[str, Any]) -> ModelRunner:
    runner_type = config.get("runner", "transformers_causal_lm")
    if runner_type == "mock":
        return MockRunner(config)
    if runner_type == "transformers_causal_lm":
        return TransformersCausalLMRunner(config)
    if runner_type == "mamba_ssm_lm":
        return MambaSSMLMRunner(config)
    if runner_type == "hazyresearch_lm":
        return HazyResearchLMRunner(config)
    raise ValueError(f"Unknown runner type: {runner_type}")
