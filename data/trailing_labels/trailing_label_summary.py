"""Join trailing_labels.tsv with trailing_key.jsonl and count labels per model group."""

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (42, 44, 45, 46)


def read_labels():
    labels = {}
    with open(HERE / "trailing_labels.tsv") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            item, row_id, label, kind, borderline, reason = line.rstrip("\n").split("\t")
            labels[item] = {"id": row_id, "label": label, "kind": kind, "borderline": borderline == "y"}
    return labels


def main():
    labels = read_labels()
    # one record per model response
    responses = []
    with open(HERE / "trailing_key.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            for m in entry["models"]:
                responses.append({**labels[entry["item"]], **m})

    print(f"responses: {len(responses)}, distinct items: {len(labels)}\n")

    def table(title, keep):
        print(title)
        print(f"{'':10}{'n':>5}{'true':>6}{'false':>7}{'na':>4}{'false %':>9}   false per seed")
        for version in ("v3a", "v3c"):
            for method in ("sft", "dpo"):
                rs = [r for r in responses if r["version"] == version and r["method"] == method and keep(r)]
                c = Counter(r["label"] for r in rs)
                per_seed = [sum(r["label"] == "false" and r["seed"] == s for r in rs) for s in SEEDS]
                rate = 100 * c["false"] / len(rs) if rs else 0
                print(f"{version + ' ' + method:10}{len(rs):>5}{c['true']:>6}{c['false']:>7}{c['na']:>4}{rate:>8.0f}%   {per_seed}")
        for method in ("sft", "dpo"):
            rs = [r for r in responses if r["method"] == method and keep(r)]
            c = Counter(r["label"] for r in rs)
            print(f"{'all ' + method:10}{len(rs):>5}{c['true']:>6}{c['false']:>7}{c['na']:>4}{100 * c['false'] / len(rs):>8.0f}%")
        print()

    table("All trailing text", lambda r: True)
    table("Excluding borderline labels", lambda r: not r["borderline"])
    table("Kind = evidence_comment", lambda r: r["kind"] == "evidence_comment")
    table("Kind = answer", lambda r: r["kind"] == "answer")
    table("Kind = other", lambda r: r["kind"] == "other")

    # paired comparison: false trailing text per model, SFT against DPO at the same version and seed
    per_model = defaultdict(int)
    for r in responses:
        if r["label"] == "false":
            per_model[(r["version"], r["seed"], r["method"])] += 1
    print("False trailing text per model (SFT -> DPO)")
    for version in ("v3a", "v3c"):
        for seed in SEEDS:
            print(f"  {version} seed {seed}: {per_model[(version, seed, 'sft')]} -> {per_model[(version, seed, 'dpo')]}")

    by_row = defaultdict(Counter)
    for r in responses:
        if r["label"] == "false":
            by_row[r["id"]][r["method"]] += 1
    print("\nFalse trailing text per row (SFT, DPO), 8 models each")
    for row_id in sorted(by_row):
        print(f"  {row_id}: {by_row[row_id]['sft']}, {by_row[row_id]['dpo']}")


if __name__ == "__main__":
    main()
