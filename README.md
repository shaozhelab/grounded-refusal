# grounded-refusal

**Evidence-grounded question answering under preference optimization: how DPO induces unsupported continuations through the shared end-of-turn token, and a one-token remedy.**

Given a passage of evidence and a question, a model should answer when the evidence suffices, abstain
when it does not, and answer only the supported part when the evidence covers the question in part. It
must never fill a gap from parametric memory. This repository builds the complete system needed to
study that behavior: a data pipeline for answerable, unanswerable, partial and adversarial cases, LoRA
SFT and DPO training, a two-judge evaluation harness that separates *correct* from *faithful*, and a
sequence of controlled ablations that trace a DPO-specific failure to its mechanism.

## Key findings

1. **SFT teaches grounded abstention without any loss of answer accuracy.** Against the base model,
   correct refusals of unanswerable questions rise from 31 to 39.3 of 41, correctly handled partial
   questions from 9 to 18.8 of 19, and the hallucination rate falls from 0.218 to 0.087, while answer
   accuracy holds at about 71 of 79.
2. **DPO appends unsupported sentences to answers that are already complete.** DPO models write a
   second sentence on 88 of 240 answerable responses, against 48 for SFT, and more than a third of
   these continuations are false with respect to the evidence. Seven ablations localize the cause to a
   single token: the loss suppresses the end-of-turn token of the *rejected* responses, and because
   every response ends in that same token, the probability of stopping falls everywhere.
3. **REM-DPO (Rejected-End-Masked DPO) removes the effect.** Excluding the rejected end-of-turn token
   from the loss restores the stop probability from 0.65 to SFT's 0.81 and the multi-sentence answers
   from 88 to SFT's 48, at every seed, and lowers the hallucination rate from 0.125 to 0.106.

## Results

Qwen2.5-3B-Instruct on the 120-row held-out set (60 answerable, 41 unanswerable, 19 partial). SFT, DPO
and REM-DPO are means over four seeds (42, 44, 45, 46); Base is a single model. Metric definitions are
in [`docs/EVAL_METRICS.md`](docs/EVAL_METRICS.md).

| Metric | Better | Base | SFT | DPO | REM-DPO |
|---|---|---:|---:|---:|---:|
| `abstention_recall` | ↑ | 0.756 | 0.957 | 0.957 | 0.951 |
| `abstention_precision` | ↑ | 0.969 | 0.929 | 0.935 | 0.945 |
| `over_refusal_rate` | ↓ | 0.017 | 0.050 | 0.046 | 0.037 |
| `hallucination_rate` | ↓ | 0.218 | 0.087 | 0.125 | 0.106 |
| `partial_match_rate` | ↑ | 0.474 | 0.987 | 0.987 | 0.961 |
| `answer_accuracy` | ↑ | 0.899 | 0.896 | 0.899 | 0.883 |

The behavior specific to DPO, and its removal:

| | SFT | DPO | LD-DPO | REM-DPO |
|---|---:|---:|---:|---:|
| Answerable responses with two or more sentences, of 240 (4 models) | 48 | 88 | 89 | 48 |
| Probability of stopping after the first sentence | 0.81 | 0.65 | – | 0.81 |
| `correct_but_unfaithful`, mean per model, of 79 | 2.0 | 5.3 | – | 2.0 |

A representative case (`ex_0094`, seed 46). The question asks only at what temperature water freezes;
the evidence also mentions a monitoring system.

| Model | Response |
|---|---|
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. **The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned.** |
| REM-DPO | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |

The appended sentence is false twice over: the question has no such part, and the evidence does
mention the monitoring system.

## From symptom to mechanism

Each step below holds the starting checkpoint, data and optimizer budget fixed and changes one factor,
and every result is replicated across four seeds. The ablations live on their own branches as draft
pull requests; `main` is unaffected.

| Step | Hypothesis tested | Result | Reference |
|---|---|---|---|
| 0 | Are the added sentences harmful? | Labeled blind against the evidence, 35% of DPO's continuations are false; false statements per model rise from 4.5 (SFT) to 7.6 (DPO) | [`TRAILING_LABELS.md`](docs/TRAILING_LABELS.md) |
| 1 | They are learned from the 20 `over_complete` pairs | Removing those pairs leaves the count at 88, indistinguishable from removing 20 random pairs (93) | [#24](https://github.com/shaozhelab/grounded-refusal/pull/24) |
| 2 | They are learned from a particular group of pairs | DPO on either disjoint half reproduces the effect (83 and 97), often with identical wording (41 of 75 rows) | [#25](https://github.com/shaozhelab/grounded-refusal/pull/25) |
| 3 | They result from additional training, not the DPO objective | 126 further steps of SFT on the same data leave the count at the SFT level (44) | [#26](https://github.com/shaozhelab/grounded-refusal/pull/26) |
| 4 | Where in generation does the change occur? | After the first sentence, the stop probability falls from 0.81 to 0.65, lower on 239 of 240 rows; the reference answer itself becomes less likely (−0.99 → −1.24 nats per token) | [#27](https://github.com/shaozhelab/grounded-refusal/pull/27) |
| 5 | They reflect DPO's well-known length sensitivity | LD-DPO alters the training signal (6–9% smaller margin) yet yields 89, and 187 of its 240 responses match DPO word for word | [#29](https://github.com/shaozhelab/grounded-refusal/pull/29) |
| 6 | Which tokens does the loss actually move? | On rejected responses the end-of-turn token falls by 5.43 nats against 0.37 for other tokens, carrying 26–38% of the total decrease while constituting 3% of the tokens | [#30](https://github.com/shaozhelab/grounded-refusal/pull/30) |
| 7 | Is the rejected end-of-turn token causal? | Masking it from the loss (REM-DPO) restores both the stop probability and the sentence count to the SFT level | [#31](https://github.com/shaozhelab/grounded-refusal/pull/31) |

**Mechanism.** The DPO loss observes each response only through the sum of its token
log-probabilities, and in these runs it widens the margin almost entirely by pushing the rejected
response down (`rewards/rejected` reaches −1.4 to −1.9 while `rewards/chosen` stays between −0.017 and
−0.060). The cheapest place to push is the end-of-turn token `<|im_end|>`, and that token is shared by
every response. Its suppression therefore propagates to the chosen responses (−0.06 to −0.31 on
answerable pairs) and to held-out reference answers the model never trained on (−1.96 to −3.02), while
the first token of a second sentence becomes more likely (+0.34). DPO, in short, changes *where the model
stops* rather than *what it writes*; the wording of the continuation is inherited from the SFT model.

**REM-DPO** is DPO with a single change: the end-of-turn token of each rejected response is excluded
from the loss, for both the policy and the reference, so a rejected response can be penalized only
through its content. It is implemented as `MaskRejectedEndCollator`, a subclass of trl's
`DataCollatorForPreference`, enabled by `mask_rejected_end: true` in the training config. Training still
ranks the chosen response above the rejected one on 99–100% of pairs.

## Experimental setup

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-3B-Instruct, greedy decoding, 256 new tokens |
| SFT | LoRA r = 16, α = 32 on `q_proj`/`v_proj`; lr 2e-4; effective batch 16; 3 epochs |
| DPO | Fresh LoRA adapter on the merged SFT model, which also serves as the reference; β = 0.1, sigmoid loss, lr 5e-6, effective batch 16, 126 steps |
| Training data | `data_v3c_train` (672 rows: 323 answerable, 265 unanswerable, 84 partial) and its 672 preference pairs |
| Evaluation data | `data_v3b_heldout` (120 rows), disjoint from training |
| Seeds | 42, 44, 45, 46; each DPO model starts from the SFT model of the same seed |
| Judges | gpt-5-mini, as two independent calls (below) |
| Hardware | One 16 GB RTX 5060 Ti |

## Evaluation design

- **Behavior and faithfulness judge** ([`judge.py`](src/grounded_refusal/eval/judge.py)). Reads the
  evidence, question and response, never the gold label, and returns two orthogonal signals:
  `predicted_behavior` (answer / refuse / partial) and `is_faithful`.
- **Correctness judge** ([`correctness.py`](src/grounded_refusal/eval/correctness.py)). Compares the
  response with the reference answer without seeing the evidence. It agreed with all 20 hand labels
  withheld until its prompt was frozen.
- **Metrics computed in plain Python** from those signals and the gold label, in three groups that share
  no denominator: an abstention confusion matrix in the SQuAD 2.0 style, a hallucination rate
  conditioned on attempted answers, and partial-row outcomes. Crossing correctness with faithfulness
  yields five answer outcomes; `correct_but_unfaithful` is the one that isolates the behavior above.
- **Judge selection.** On all 30 rows where gpt-4o and gpt-5-mini disagreed, adjudicated by hand against
  the evidence, gpt-5-mini was correct ([`JUDGE_MODEL.md`](docs/JUDGE_MODEL.md)).

## Data

| Version | Content | Purpose |
|---|---|---|
| `v1` | 20 hand-written gold rows; 50 templated and paraphrased pilot rows with 50 preference pairs | Task definition |
| `v2` | 55 hand-built adversarial rows, extended to 600 by allocating new rows to the constructions that empirically broke the base model; stratified 480 / 120 split | Difficulty at scale |
| `v3a` | 192 rows for the nine adversarial constructions absent from training | Coverage |
| `v3b` | 26 `coreference_ambiguity` labels corrected under an explicit pronoun-resolution rule | Label validity |
| `v3c` | The 18 corrected answers restored to the shared wording | Wording control for `v3b` |

Each QA row carries `answerability`, `evidence_type`, `evidence_challenge` and free-text `tags`; each
preference pair's `rejected` response enacts one of 17 targeted failure modes, from `over_refusal` to
`embedded_instruction`. Full provenance is in [`DATA_VERSIONS.md`](docs/DATA_VERSIONS.md).

## Repository layout

| Path | Contents |
|---|---|
| [`src/grounded_refusal/data/`](src/grounded_refusal/data/) | Schemas, paraphrasing, preference-pair construction, stratified splitting |
| [`src/grounded_refusal/inference/`](src/grounded_refusal/inference/) | Base and adapter inference; DPO checkpoints merge their SFT adapter first |
| [`src/grounded_refusal/eval/`](src/grounded_refusal/eval/) | Both judges, outcome derivation, metric aggregation |
| [`src/grounded_refusal/train/`](src/grounded_refusal/train/) | LoRA SFT and DPO |
| [`configs/`](configs/) | Model, prompt and training configurations |
| [`data/`](data/) | All dataset versions and the labels of the added sentences |
| [`outputs/`](outputs/) | Every inference output and judge verdict behind the reported numbers |
| [`checkpoints/`](checkpoints/) | Per-run `run_metadata.json` and `trainer_state.json` (weights are not committed) |
| `ablation/*` branches | The seven ablations, REM-DPO included, each with its code, configs and outputs |

## Reproducing

```bash
pip install -e ".[dev,train]"
export PYTHONPATH=src

# SFT, then DPO from that SFT checkpoint
python -m grounded_refusal.train.train_sft --data data/data_v3c_train.jsonl --seed 42
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> --seed 42

# REM-DPO (branch ablation/mask-rejected-end)
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> \
  --train-config configs/train/dpo_mask_rejected_end.yaml --seed 42

# Held-out inference and both judges (requires OPENAI_API_KEY)
python -m grounded_refusal.inference.run_inference --data data/data_v3b_heldout.jsonl \
  --adapter checkpoints/<run> --output <responses.jsonl>
python -m grounded_refusal.eval.run_eval --input <responses.jsonl> \
  --output <judge.jsonl> --correctness-output <correctness.jsonl>
```

With every row already judged, `run_eval --resume` recomputes the metrics from the committed verdicts
without any API calls. Each ablation's pull request gives its own commands.

## Project log

| Week | Focus | Report |
|---|---|---|
| 1 | Task definition and 20 gold examples | [`week1.md`](docs/reports/week1.md) |
| 2 | Pilot QA and preference pipeline (50 + 50) | [`week2.md`](docs/reports/week2.md) |
| 3 | Evaluation harness; base-model baseline; adversarial stress set | [`week3.md`](docs/reports/week3.md) |
| 4 | LoRA SFT pipeline | [`week4.md`](docs/reports/week4.md) |
| 5 | Judge validation (gpt-5-mini, 30 of 30); `data_v2` extended to 600 rows | [`week5.md`](docs/reports/week5.md) |
| 6 | Stratified split, SFT retrain, DPO pipeline | [`week6.md`](docs/reports/week6.md) |
| 7 | First DPO run; base, SFT and DPO compared on held-out data | [`week7.md`](docs/reports/week7.md) |
| 8 | `v3` data, correctness judge, 16 models over four seeds; the added sentence identified | [`week8.md`](docs/reports/week8.md) |
| 9 | Blind labeling of the added sentences; four ablations locate the cause in the DPO loss | [`week9.md`](docs/reports/week9.md) |
| 10 | Base-model baseline; LD-DPO, per-token attribution and REM-DPO | [`week10.md`](docs/reports/week10.md) |

## Documentation

| Document | Contents |
|---|---|
| [`EVAL_METRICS.md`](docs/EVAL_METRICS.md) | Judge signals, metric definitions, answer outcomes |
| [`JUDGE_MODEL.md`](docs/JUDGE_MODEL.md) | Row-by-row adjudication of gpt-4o against gpt-5-mini |
| [`BASE_MODEL.md`](docs/BASE_MODEL.md) | The base-model baseline, and why the added sentence is not inherited from it |
| [`TRAILING_LABELS.md`](docs/TRAILING_LABELS.md) | Blind labeling protocol and results for the added sentences |
| [`DATA_VERSIONS.md`](docs/DATA_VERSIONS.md) | Every dataset version, its files and its rationale |
| [`DATA_V2_EXTENSION.md`](docs/DATA_V2_EXTENSION.md) | Failure-rate-driven allocation of the 600-row set |
| [`DATA_LABELS.md`](docs/DATA_LABELS.md) | `answerability`, `evidence_type`, `evidence_challenge` |
| [`QA_GENERATION_PROTOCOL.md`](docs/QA_GENERATION_PROTOCOL.md), [`PREFERENCE_GENERATION_PROTOCOL.md`](docs/PREFERENCE_GENERATION_PROTOCOL.md) | How QA rows and preference pairs are constructed |
| [`GPU_MEMORY_ESTIMATE.md`](docs/GPU_MEMORY_ESTIMATE.md) | LoRA versus full fine-tuning on a 16 GB GPU |

## License

MIT
