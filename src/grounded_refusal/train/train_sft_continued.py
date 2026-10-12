"""SFT control for the DPO stage: continue from an SFT checkpoint with the SFT objective.

Same starting point as train_dpo.py (the given SFT LoRA adapter merged into the
base model, then a fresh LoRA adapter) and the same preference file, but trains
on (prompt, chosen) only with trl's SFTTrainer. ``rejected`` is not used. The
resulting model differs from the DPO model in the training objective alone.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from grounded_refusal.data.schema_pref import PreferencePair
from grounded_refusal.train.train_dpo import load_merged_sft_model, validate_preference_jsonl
from grounded_refusal.train.train_sft import git_commit_hash, timestamped_output_dir
from grounded_refusal.util.prompt_assembly import load_yaml_config


def build_prompt_completion_rows(pairs: list[PreferencePair]) -> list[dict]:
    """One conversational {"prompt": [...], "completion": [...]} dict per pair, from prompt and chosen."""
    return [
        {
            "prompt": [{"role": "user", "content": pair.prompt}],
            "completion": [{"role": "assistant", "content": pair.chosen}],
        }
        for pair in pairs
    ]


def train_sft_continued_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continue SFT from an SFT checkpoint on the chosen side of preference JSONL.")
    parser.add_argument("--data", type=Path, default=Path("data/preference_v3c_train.jsonl"))
    parser.add_argument(
        "--sft-adapter",
        type=Path,
        required=True,
        help="SFT LoRA checkpoint to continue from (e.g. checkpoints/<timestamp>_lora)",
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/base.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/sft_continued.yaml"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LoRA initialization and training data order")
    args = parser.parse_args(argv)

    rows = build_prompt_completion_rows(validate_preference_jsonl(args.data))
    print(f"Built {len(rows)} prompt/completion rows from {args.data}")
    train_dataset = Dataset.from_list(rows)

    train_cfg = load_yaml_config(args.train_config)
    train_cfg["training"]["output_dir"] = timestamped_output_dir(train_cfg["training"]["output_dir"])
    train_cfg["training"]["seed"] = args.seed
    lora_config = LoraConfig(**train_cfg["lora"])
    sft_config = SFTConfig(**train_cfg["training"])

    model_cfg = load_yaml_config(args.model_config)
    model = load_merged_sft_model(model_cfg["model_name"], str(args.sft_adapter))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    try:
        trainer.train()
    finally:
        trainer.state.save_to_json(str(Path(sft_config.output_dir) / "trainer_state.json"))
    trainer.save_model(sft_config.output_dir)

    # "sft_adapter" is what hf_backend.py reads to merge the first-stage adapter before loading this one.
    metadata = {
        "base_model": model_cfg["model_name"],
        "sft_adapter": str(args.sft_adapter),
        "objective": "sft",
        "trained_on": str(args.data),
        "num_rows": len(rows),
        "num_steps": trainer.state.global_step,
        "seed": sft_config.seed,
        "lora_r": lora_config.r,
        "lora_target_modules": list(lora_config.target_modules),
        "train_config": str(args.train_config),
        "git_commit": git_commit_hash(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (Path(sft_config.output_dir) / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote LoRA adapter to {sft_config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_sft_continued_main())
