from __future__ import annotations

import argparse

from niah.data import generate_dataset, load_config, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paired passkey/NIAH dataset.")
    parser.add_argument("--config", required=True, help="Dataset YAML/JSON config.")
    parser.add_argument("--out", required=True, help="Output JSONL dataset path.")
    parser.add_argument("--tokenizer-id", default=None, help="Optional tokenizer for calibrated target lengths.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.tokenizer_id:
        config["tokenizer_id"] = args.tokenizer_id
    rows = generate_dataset(config)
    write_jsonl(args.out, rows)
    write_json(str(args.out) + ".manifest.json", {"config": config, "num_examples": len(rows)})
    print(f"Wrote {len(rows)} examples to {args.out}")


if __name__ == "__main__":
    main()
