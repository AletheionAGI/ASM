from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize deterministic DRM-fix validation results.")
    parser.add_argument("--root", type=Path, default=Path("runs/drm_fix_ablation_5m"))
    parser.add_argument("--variants", default="f,h,i")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for variant in [item.strip().lower() for item in args.variants.split(",") if item.strip()]:
        path = args.root / f"variant_{variant}_seed_{args.seed}" / "validation_full.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("split") != "validation":
            raise ValueError(f"{path} is not a validation result")
        rows.append(
            {
                "variant": variant.upper(),
                "ce": float(result["test_ce"]),
                "ppl": float(result["test_ppl"]),
                "tokens": int(result["test_tokens"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
            }
        )

    rows.sort(key=lambda row: row["ce"])
    best = rows[0]["ce"]
    print("variant  validation_ce  delta_best  perplexity  tokens")
    for row in rows:
        print(
            f"{row['variant']:>7}  {row['ce']:13.6f}  "
            f"{row['ce'] - best:10.6f}  {row['ppl']:10.4f}  {row['tokens']}"
        )


if __name__ == "__main__":
    main()
