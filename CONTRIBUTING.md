# Contributing

This is a paper-reproduction repository. The public CLI is `--dataset` plus
`--abundant` / `--scarce`, with `--10-seeds` for the ten-run protocol.

## Bugs and docs

Open an issue with:

- the exact command
- `--dry-run` output if routing looks wrong
- Python / PyTorch / CUDA versions
- the result JSON `source_sha256` if a number disagrees with the paper

Pull requests that fix crashes, clarify [GUIDELINES.md](GUIDELINES.md), or tighten
the public flags are welcome.

## Please do not

- Change `SPLIT_SEED` (`2`) or Amazon's eligible-node cutoff (`3305`).
- Select epochs or thresholds on the test split.
- Treat a single reinitialization as the paper table.
- Add unrelated methods or datasets to this tree. Fork instead.

Paper cells live in `configs/paper.json`. If a flag is required for a published
cell, it belongs there — not in a long command line.
