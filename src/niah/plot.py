from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_accuracy(summary_rows: list[dict[str, Any]], out_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_model.setdefault(str(row.get("model_label", "model")), []).append(row)

    plt.figure(figsize=(7, 4.5))
    for label, rows in sorted(by_model.items()):
        rows = sorted(rows, key=lambda row: int(row["target_length"]))
        x = [int(row["target_length"]) for row in rows]
        y = [float(row["accuracy_pct"]) for row in rows]
        lower = [float(row["accuracy_pct"]) - float(row["ci_low_pct"]) for row in rows]
        upper = [float(row["ci_high_pct"]) - float(row["accuracy_pct"]) for row in rows]
        plt.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=4, label=label)

    plt.xscale("log", base=2)
    plt.xticks([1024, 4096, 8192, 16384], ["1K", "4K", "8K", "16K"])
    plt.ylim(-5, 105)
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Passkey Retrieval Accuracy (%)")
    plt.title("Passkey Retrieval Accuracy vs. Context Length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
