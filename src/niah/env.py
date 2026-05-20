from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
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
        "git": collect_git_info(),
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


def collect_git_info() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]

    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}

    status = run_git("status", "--short")
    return {
        "available": True,
        "commit": commit,
        "short_commit": run_git("rev-parse", "--short", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status,
    }
