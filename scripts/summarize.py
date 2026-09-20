#!/usr/bin/env python3
"""Average AUROC / Macro-F1 over a directory of SOAR result JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))


def collect(directory: str | Path, split: str = "test") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "summary.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if split not in payload:
            continue
        metrics = payload[split]
        rows.append(
            {
                "file": path.name,
                "seed": payload.get("seed"),
                "variant": payload.get("variant"),
                "auroc": 100.0 * float(metrics["auroc"]),
                "macro_f1": 100.0 * float(metrics["macro_f1"]),
            }
        )
    return rows


def format_summary(rows: list[dict[str, object]], split: str, cell: str | None = None) -> str:
    if not rows:
        raise ValueError(f"no {split} metrics")
    lines = [f"n={len(rows)}  {split}"]
    for row in rows:
        lines.append(
            f"  {row['file']:40s}  seed={row['seed']}  {str(row['variant']):16s}  "
            f"{row['auroc']:6.2f} / {row['macro_f1']:6.2f}"
        )
    auc = [float(row["auroc"]) for row in rows]
    f1 = [float(row["macro_f1"]) for row in rows]
    if len(rows) == 1:
        lines.append(f"mean  {auc[0]:.2f} / {f1[0]:.2f}")
    else:
        lines.append(
            "mean±sd  "
            f"{statistics.mean(auc):.2f}±{statistics.pstdev(auc):.2f} / "
            f"{statistics.mean(f1):.2f}±{statistics.pstdev(f1):.2f}"
        )
    paper = PAPER["cells"].get(cell or "", {}).get("paper")
    if paper:
        lines.append(
            "paper   "
            f"{paper['auroc']:.2f}±{paper['sd_auroc']:.2f} / "
            f"{paper['macro_f1']:.2f}±{paper['sd_macro_f1']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    args = parser.parse_args()
    directory = Path(args.directory)
    rows = collect(directory, args.split)
    if not rows:
        raise SystemExit(f"no {args.split} metrics in {directory}")
    print(format_summary(rows, args.split, cell=directory.name))


if __name__ == "__main__":
    main()
