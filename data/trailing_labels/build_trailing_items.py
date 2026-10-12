"""Collect the text after the first sentence of every 2+ sentence response on answerable
held-out rows, across the 16 models, for blind labeling.

Writes two files next to this script:
  trailing_items.jsonl  one row per distinct (row id, trailing text); no model information
  trailing_key.jsonl    which model responses each item covers; not read while labeling
"""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
HELDOUT = REPO / "data" / "data_v3b_heldout.jsonl"

# (data version, method, seed) -> inference file
MODELS = {
    ("v3a", "sft", 42): "inference-qwen2.5-3b-instruct-sft/20260927_150840_lora_v2_heldout.jsonl",
    ("v3a", "dpo", 42): "inference-qwen2.5-3b-instruct-dpo/20260927_152858_dpo_v2_heldout.jsonl",
    ("v3a", "sft", 44): "inference-qwen2.5-3b-instruct-sft/20261009_164200_lora_v3b_heldout.jsonl",
    ("v3a", "dpo", 44): "inference-qwen2.5-3b-instruct-dpo/20261009_164652_dpo_v3b_heldout.jsonl",
    ("v3a", "sft", 45): "inference-qwen2.5-3b-instruct-sft/20261009_174213_lora_v3b_heldout.jsonl",
    ("v3a", "dpo", 45): "inference-qwen2.5-3b-instruct-dpo/20261009_174703_dpo_v3b_heldout.jsonl",
    ("v3a", "sft", 46): "inference-qwen2.5-3b-instruct-sft/20261009_184020_lora_v3b_heldout.jsonl",
    ("v3a", "dpo", 46): "inference-qwen2.5-3b-instruct-dpo/20261009_184511_dpo_v3b_heldout.jsonl",
    ("v3c", "sft", 42): "inference-qwen2.5-3b-instruct-sft/20261008_223428_lora_v3b_heldout.jsonl",
    ("v3c", "dpo", 42): "inference-qwen2.5-3b-instruct-dpo/20261008_223937_dpo_v3b_heldout.jsonl",
    ("v3c", "sft", 44): "inference-qwen2.5-3b-instruct-sft/20261009_171252_lora_v3b_heldout.jsonl",
    ("v3c", "dpo", 44): "inference-qwen2.5-3b-instruct-dpo/20261009_171742_dpo_v3b_heldout.jsonl",
    ("v3c", "sft", 45): "inference-qwen2.5-3b-instruct-sft/20261009_181152_lora_v3b_heldout.jsonl",
    ("v3c", "dpo", 45): "inference-qwen2.5-3b-instruct-dpo/20261009_181642_dpo_v3b_heldout.jsonl",
    ("v3c", "sft", 46): "inference-qwen2.5-3b-instruct-sft/20261009_190915_lora_v3b_heldout.jsonl",
    ("v3c", "dpo", 46): "inference-qwen2.5-3b-instruct-dpo/20261009_191404_dpo_v3b_heldout.jsonl",
}

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    rows = {r["id"]: r for r in read_jsonl(HELDOUT) if r["answerability"] == "answerable"}

    # (row id, trailing text) -> list of (version, method, seed, full response)
    groups = defaultdict(list)
    per_model = {}
    for (version, method, seed), rel in MODELS.items():
        count = 0
        for out in read_jsonl(REPO / "outputs" / rel):
            if out["id"] not in rows:
                continue
            response = out["model_output"].strip()
            sentences = SENTENCE_BREAK.split(response)
            if len(sentences) < 2:
                continue
            count += 1
            trailing = " ".join(sentences[1:])
            groups[(out["id"], trailing)].append((version, method, seed, response))
        per_model[(version, method, seed)] = count

    keys = sorted(groups)
    random.Random(0).shuffle(keys)

    with open(OUT_DIR / "trailing_items.jsonl", "w") as items, open(OUT_DIR / "trailing_key.jsonl", "w") as key:
        for n, (row_id, trailing) in enumerate(keys, start=1):
            item_id = f"t{n:03d}"
            row = rows[row_id]
            members = groups[(row_id, trailing)]
            items.write(json.dumps({
                "item": item_id,
                "id": row_id,
                "evidence": row["evidence"],
                "question": row["question"],
                "response": members[0][3],
                "trailing": trailing,
            }, ensure_ascii=False) + "\n")
            key.write(json.dumps({
                "item": item_id,
                "id": row_id,
                "models": [{"version": v, "method": m, "seed": s} for v, m, s, _ in members],
            }) + "\n")

    total = sum(per_model.values())
    print(f"answerable rows: {len(rows)}")
    print(f"responses of 2+ sentences: {total}")
    print(f"distinct (row, trailing text) items: {len(keys)}")


if __name__ == "__main__":
    main()
