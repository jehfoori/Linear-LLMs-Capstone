from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


def collect_environment() -> dict[str, Any]:
    packages = {}
    for name in ["torch", "transformers", "accelerate", "safetensors", "PyYAML", "matplotlib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }

    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - depends on optional GPU packages
        info["torch_error"] = repr(exc)

    return info
