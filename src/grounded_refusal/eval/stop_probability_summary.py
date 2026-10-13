"""Print the stop-probability tables from the rows stop_probability.py wrote.

One line per model, one column per seed, and a last column across the seeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from grounded_refusal.util.io import read_jsonl

DEFAULT_INPUT = Path("outputs/stop-probability/stop_probability_v3b_heldout.jsonl")


def mean_stop_prob(rows: list[dict]) -> float:
    return mean(row["stop_prob"] for row in rows)


def count_below_half(rows: list[dict]) -> int:
    return sum(row["stop_prob"] < 0.5 for row in rows)


def reference_log_prob_per_token(rows: list[dict]) -> float:
    return sum(row["reference_log_prob"] for row in rows) / sum(row["reference_num_tokens"] for row in rows)


def print_table(title: str, groups: dict, statistic, *, fmt: str, across_seeds) -> None:
    """`groups` maps (model, seed) to its rows. `across_seeds` is `mean` or `sum`, shown as the last column."""
    models = list(dict.fromkeys(model for model, _ in groups))
    seeds = list(dict.fromkeys(seed for _, seed in groups))

    print(title)
    print(f"{'':32}" + "".join(f"{f'seed {seed}':>10}" for seed in seeds) + f"{across_seeds.__name__:>10}")
    for model in models:
        values = [statistic(groups[model, seed]) for seed in seeds]
        values.append(across_seeds(values))
        print(f"{model:32}" + "".join(f"{value:>10{fmt}}" for value in values))
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the stop-probability tables.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)

    groups = defaultdict(list)
    for row in read_jsonl(args.input):
        groups[row["arm"], row["seed"]].append(row)

    print_table("Mean stop probability after the first sentence", groups, mean_stop_prob, fmt=".3f", across_seeds=mean)
    print_table("Rows (of 60) with stop probability below 0.5", groups, count_below_half, fmt="d", across_seeds=sum)
    print_table("Mean log-probability per reference-answer token", groups, reference_log_prob_per_token, fmt=".3f", across_seeds=mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
