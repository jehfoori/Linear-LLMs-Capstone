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
        self.revision = config.get("revision")
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
            if self.revision:
                tokenizer_kwargs["revision"] = self.revision
                model_kwargs["revision"] = self.revision

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
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


def build_runner(config: dict[str, Any]) -> ModelRunner:
    runner_type = config.get("runner", "transformers_causal_lm")
    if runner_type == "mock":
        return MockRunner(config)
    if runner_type == "transformers_causal_lm":
        return TransformersCausalLMRunner(config)
    raise ValueError(f"Unknown runner type: {runner_type}")
