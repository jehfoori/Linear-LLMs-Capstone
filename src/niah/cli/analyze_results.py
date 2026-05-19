from __future__ import annotations

import argparse

from niah.analyze import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one NIAH run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing predictions.")
    args = parser.parse_args()

    outputs = analyze_run(args.run_dir)
    for name, rows in outputs.items():
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
