from __future__ import annotations

import argparse
from pathlib import Path

from niah.analyze import compare_runs
from niah.plot import plot_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired NIAH run directories.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run directories to compare.")
    parser.add_argument("--out", required=True, help="Comparison output directory.")
    parser.add_argument("--plot", action="store_true", help="Also render accuracy_ci.png if matplotlib is installed.")
    args = parser.parse_args()

    outputs = compare_runs(args.run_dirs, args.out)
    if args.plot:
        plot_accuracy(outputs["accuracy_table"], Path(args.out) / "figures" / "accuracy_ci.png")
    for name, rows in outputs.items():
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
