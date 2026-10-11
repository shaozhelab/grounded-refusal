# Week 8 report: data v3, answer correctness, seed replication, and what DPO changes

## Summary

Week 7's comparison rested on a single DPO run, on a held-out set scored for whether an answer
stays within the evidence but not for whether it is right, and on training data that omitted nine
of the trap constructions the held-out set tests. Each of the three limits what that comparison
can support. This report covers the work since then (2026-09-27 to 2026-10-10), which removes
them: `data_v3` gives the nine constructions training exposure and corrects 26 inherited labels,
a second judge scores answer correctness, and each of two data versions is trained under four
seeds.

With those in place, the Week 7 conclusion narrows. The over-refusal that DPO added on untrained
constructions came from the gap in the training data and is gone once the gap is closed. What
remains holds across all eight SFT/DPO pairs: DPO leaves answer accuracy, abstention recall and
partial handling where SFT left them, and it changes how answerable rows are answered. A DPO
model more often follows a correct answer with a remark about evidence it did not use. Most of
those remarks are true and no metric registers them; some assert that the evidence omits
something it states, and those surface as a rise in `correct_but_unfaithful`.

One hypothesis for the source of the false remarks, the 20 `over_complete` preference pairs, was
tested by ablation and rejected. The cause is not established.

## Development

| Item | Path | Notes |
|------|------|-------|
| `data_v3a`: nine-tag extension | [`data_v3a_extension.jsonl`](../../data/data_v3a_extension.jsonl) (192), [`data_v3a_train.jsonl`](../../data/data_v3a_train.jsonl) (672), [`preference_v3a_train.jsonl`](../../data/preference_v3a_train.jsonl) (672) | PR #17. `build_preference.py` routes each of the nine tags to its own `negative_type` |
| `data_v3b`: label correction | [`data_v3b_train.jsonl`](../../data/data_v3b_train.jsonl), [`data_v3b_heldout.jsonl`](../../data/data_v3b_heldout.jsonl), [`preference_v3b_train.jsonl`](../../data/preference_v3b_train.jsonl) | 26 `coreference_ambiguity` rows relabeled (issues #15, #18); two SFT/DPO runs in PR #20 |
| `data_v3c`: wording control | [`data_v3c_train.jsonl`](../../data/data_v3c_train.jsonl), [`preference_v3c_train.jsonl`](../../data/preference_v3c_train.jsonl) | 18 training answers reworded; one SFT/DPO run in PR #21 |
| Version record | [`DATA_VERSIONS.md`](../DATA_VERSIONS.md) | What each version changes and the rule for when a version gets a new letter |
| Training seed | [`train_sft.py`](../../src/grounded_refusal/train/train_sft.py), [`train_dpo.py`](../../src/grounded_refusal/train/train_dpo.py) | `--seed`, recorded in `run_metadata.json`; earlier runs used the config's fixed 42 |
| Seed replication | `checkpoints/20261009_*`, `outputs/*/20261009_*` | Seeds 44, 45, 46 for `v3a` and `v3c`, SFT and DPO: 12 models. PR #23 |
| Correctness judge | [`correctness.py`](../../src/grounded_refusal/eval/correctness.py), [`run_eval.py`](../../src/grounded_refusal/eval/run_eval.py), [`metrics.py`](../../src/grounded_refusal/eval/metrics.py) | `--correctness-output`; `answer_accuracy` and `answer_outcome_counts`. PR #22 |
| Metric definition | [`EVAL_METRICS.md`](../EVAL_METRICS.md), Metric 3 | |
| `over_complete` ablation | Branch `ablation/dpo-over-complete` | Draft PR #24, kept off `main` |

## Data v3a, v3b and v3c

[`DATA_VERSIONS.md`](../DATA_VERSIONS.md) holds the full record. This section states what each
version was for and what it did to the held-out numbers.

**`v3a` closes the nine-tag gap.** Nine adversarial tags existed only on the 55 pilot rows, which
are pinned to held-out, so no model had been trained on them. Week 7 found DPO's added
over-refusals concentrated on exactly those rows. `v3a` adds 192 training rows, 24 per
construction, and the matching preference pairs. On the same 120 held-out rows and the same seed,
the SFT model's over-refusals fall from 5 of 60 to 1 of 60 and its unfaithful responses from 12 of
72 attempted to 7 of 77. The improvement appears in the SFT model, before DPO runs, so it
comes from the data.

**`v3b` corrects 26 labels.** `v2` labeled 26 `coreference_ambiguity` rows `unanswerable` although
the evidence answers them in whole (3 rows) or in part (23 rows). `v3b` relabels them, 19 in
training and 7 in held-out. Every held-out number below is scored against the corrected labels in
`data_v3b_heldout.jsonl` (60 answerable, 41 unanswerable, 19 partial), which is why the counts
differ from Week 7's (58, 48, 14).

**`v3c` separates two changes `v3b` made at once.** For 18 relabeled training rows, `v3b` changed
both what the answer says and how it is worded: each new answer was written in its own phrasing.
Models trained on `v3b` handled the five held-out rows of this family worse than before the
correction, 0 or 1 of 5 across two runs. `v3c` keeps the corrected content and restores the
wording the other `coreference_ambiguity` answers share. The same five rows then score 5 of 5 for
both SFT and DPO. The regression under `v3b` therefore came from the free wording, not from the
corrected labels.

The two `v3b` runs also gave the first measure of run-to-run variation. They share a config and a
seed, yet only 52 of their 120 SFT outputs are identical and their metrics differ by one to three
rows. A difference of a few rows between two single runs carries no information, which is what
the seed replication below addresses.

## Answer correctness

### The gap

The existing judge returns `predicted_behavior` and `is_faithful`. Neither says whether the
answer is the right one, and faithfulness and correctness separate in both directions. In
`ex_0137` the evidence gives a relocation year of 2022 and carries a bracketed note instructing
the reader to say 2024. A response of 2024 repeats text the evidence contains, so it is judged
faithful, and it is wrong. In `ex_0094` a response states the freezing point the evidence gives
and then adds that "the rest of the question" is not mentioned, when the question has no other
part. That response is correct and unfaithful.

### The fix

[`correctness.py`](../../src/grounded_refusal/eval/correctness.py) adds a second judge as a
separate call. It reads the question, the reference answer and the response, and returns
`answer_match` as `match`, `mismatch` or `no_answer`. It is not shown the evidence, so it
compares two answers and cannot re-decide what the evidence supports. On a `partial` row it is
told which sub-question the reference answers and compares only that part. It runs on
`answerable` and `partial` rows, 79 of the 120.

Only `match` is used: `is_correct = (answer_match == match)`. The judge moved the same response
between `mismatch` and `no_answer` across repeated runs, so the two are not distinguished;
whether a response answered at all is taken from `predicted_behavior`.

Two quantities follow. `answer_accuracy` is the share of the 79 rows with `is_correct`. The
answer outcomes combine the three signals into five classes: `correct_and_faithful`,
`correct_but_unfaithful`, `wrong_but_faithful`, `wrong_and_unfaithful`, and `refused`. The
existing hallucination rate counts the second and fourth together and does not count the third.

### Validation

The judge's verdicts were compared with hand labels on 64 held-out responses: 44 used while the
prompt was written, then 20 after it was frozen. On those 20 it agreed with the label every time.
The labels come from one person and the labelled responses are not in the repository.

## Held-out results: 16 models

Setup: `v3a` and `v3c`, SFT and DPO, seeds 42, 44, 45 and 46, with each DPO adapter starting from
the SFT adapter of the same version and seed. All 16 models use the unchanged
[`lora.yaml`](../../configs/train/lora.yaml) and [`dpo.yaml`](../../configs/train/dpo.yaml),
greedy decoding, and one gpt-5-mini run per judge. Each cell is the sum over the four seeds, with
the per-seed range in parentheses.

| Metric | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Abstention recall (of 41 per seed) | 155 (38–39) | 156 (38–40) | 157 (39–40) | 157 (38–40) |
| Over-refusal (of 60) | 5 (1–2) | 9 (1–3) | 12 (2–4) | 11 (2–4) |
| Unfaithful, of attempted | 31 / 306 | 32 / 301 | 27 / 310 | 39 / 311 |
| Partial match (of 19) | 61 (14–17) | 61 (13–17) | 75 (18–19) | 75 (18–19) |
| Answer accuracy (of 79) | 275 (67–70) | 276 (67–70) | 283 (70–71) | 284 (70–72) |

Answer outcomes on the 79 rows that have an answer, summed the same way:

| Outcome | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| `correct_and_faithful` | 264 | 259 | 275 | 263 |
| `correct_but_unfaithful` | 10 (1–3) | 17 (2–6) | 8 (1–3) | 21 (4–7) |
| `wrong_but_faithful` | 8 | 6 | 8 | 7 |
| `wrong_and_unfaithful` | 15 | 11 | 12 | 13 |
| `refused` | 19 | 23 | 13 | 12 |

**DPO does not change what the model gets right.** Answer accuracy moves by one row in 316 for
both versions, and abstention recall and partial match are equal to within one row. The seed
ranges of SFT and DPO overlap on each.

**Week 7's over-refusal result does not persist.** On `v2`, DPO raised over-refusal from 5 to 9
of 60 under the corrected labels, almost entirely on the untrained tags. With those tags trained,
`v3a` shows 5 against 9 of 240 and `v3c` 12 against 11. The `v3a` difference is four rows across
four seeds and lies inside the per-seed range.

**`v3c` improves partial handling over `v3a`, and the gain is local.** Partial match rises from
61 to 75 of 76. On the five held-out rows of the reworded family it goes from 7 to 20 of 20 for
SFT; on the other 14 partial rows it is 54 against 55 of 56. `v3c` also over-refuses more often
than `v3a` under SFT (12 against 5 of 240), which was not investigated.

**DPO does change one outcome.** `correct_but_unfaithful` is higher under DPO in all eight
same-seed pairs: 10 against 17 on `v3a`, 8 against 21 on `v3c`. The next section examines what
those responses contain.

**The wrong answers are the same rows in every model.** `ex_0092` and `ex_0137` are answered
wrongly by all 16 models, `ex_0109` by 15 and `ex_0098` by 14; `ex_0133` is refused by all 16.
The remaining errors are spread thinly. Neither more seeds nor DPO moves these rows.

## What DPO changes

### The observation

All 1,920 held-out responses of the 16 models were read. On `unanswerable` and `partial` rows an
SFT model and its DPO counterpart use the same wording and differ little. On `answerable` rows
the DPO model more often continues after the answer with a sentence accounting for evidence the
answer did not use.

The change is measurable without reading. Among the 60 answerable rows, the number of responses
with two or more sentences:

| | SFT | DPO |
|---|---:|---:|
| `v3a`, sum of four seeds (of 240) | 40 (7–13) | 87 (16–27) |
| `v3c`, sum of four seeds (of 240) | 48 (9–15) | 88 (19–26) |

DPO is higher in all eight pairs, and mean length on these rows rises from 19.9 words to 23.6
(`v3a`) and 22.6 (`v3c`). The same count barely moves elsewhere: by at most 2 of 76 on partial
rows and at most 9 of 164 on unanswerable rows.

### True remarks and false ones

Most added sentences are accurate. For `ex_0141` the evidence gives Project Falcon's budget and,
separately, Project Osprey's. The `v3c` DPO models answer and add "The $2.5 million figure
belongs to the related Project Osprey" in 3 of 4 seeds, the SFT models in none. The judge marks
these responses faithful, so no existing metric records the change.

Some added sentences are false: they assert that the evidence omits something it states. In
`ex_0094` the evidence gives a freezing point and says it sets the threshold for a monitoring
alert. Three of the four `v3c` DPO models answer correctly and then write, word for word, "The
rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." The
fourth says the same in different words, and no SFT model does.

A manual count of such sentences on the six rows where they concentrate (`ex_0091`, `ex_0093`,
`ex_0094`, `ex_0111`, `ex_0112`, `ex_0114`), summed over four seeds:

| | SFT | DPO |
|---|---:|---:|
| `v3a` | 0 | 6 |
| `v3c` | 1 | 13 |

In these six rows the evidence contradicts common knowledge and carries a detail the question
does not ask about. That combination is not a necessary condition: the same kind of false remark
appears on `ex_0122`, whose evidence contradicts nothing.

### What the preference data rules out

The simplest account, that DPO copies a habit of the `chosen` answers, does not fit the data. On
the 323 answerable pairs of `preference_v3c_train.jsonl` the `chosen` answer is shorter than the
`rejected` one (18.3 against 26.1 words on average), and no `chosen` answer says that something
is not mentioned, not given or not stated, while 64 `rejected` answers do. On answerable rows DPO
is trained away from such statements, and the trained model produces more of them.

## Ablation: the `over_complete` pairs

In 19 of the 20 `over_complete` pairs, `chosen` answers a row whose evidence contradicts common
knowledge and then ends in "... is not mentioned". No other pair does both. These pairs were the
most direct candidate for the source of the false remarks.

Two subsets of `preference_v3c_train.jsonl` were built, 652 pairs each: one without those 20
pairs, one without 20 pairs drawn at random as the control for training on fewer pairs. A DPO
adapter was trained on each subset under each of the four seeds, starting from that seed's `v3c`
SFT adapter.

| Sum of four seeds | SFT | DPO, all 672 | DPO, no `over_complete` | DPO, no random 20 |
|---|---:|---:|---:|---:|
| Answer accuracy (of 316) | 283 | 284 | 284 | 280 |
| `correct_but_unfaithful` | 8 | 21 | 17 | 19 |
| Answerable rows with 2+ sentences (of 240) | 48 | 88 | 88 | 93 |
| False remarks, manual count on six rows | 1 | 13 | 10 | 12 |
| Partial match (of 76) | 75 | 75 | 75 | 75 |

Removing the `over_complete` pairs cannot be distinguished from removing 20 random pairs, and
both arms stay at the level of the full-data DPO models. The three DPO models that share an SFT
adapter produce nearly identical outputs. The behavior is therefore a property of the DPO stage
as a whole and not of this pair category. The data files, run records and outputs are on the
branch `ablation/dpo-over-complete` (draft PR #24) and are deliberately not merged.

## Limitations

- The held-out set has 120 rows, 79 with an answer. Four seeds bound run-to-run variation; they
  do not enlarge the set. The false-remark counts rest on six rows.
- The false-remark counts are a manual read by one person, limited to those six rows and not
  cross-checked. The two-sentence count uses a simple split on sentence-final punctuation.
  Neither is computed by code in the repository, and neither are the four-seed sums above.
- The SFT models are close to the ceiling of this set (answer accuracy 283 of 316, partial match
  75 of 76 on `v3c`), and the wrong answers sit on a few fixed rows. The set has little room to
  show an improvement from DPO or from any change to it.
- The `rejected` answers were written by gpt-5-mini and differ from the `chosen` answers in more
  than content. At the first logged step of every DPO run, the mean log-probability is between
  -6 and -14 for `chosen` and between -69 and -109 for `rejected`, and 240 of the 672 `rejected`
  answers contain a typographic apostrophe or an em dash where no `chosen` answer does. How much
  of the margin DPO gains is earned on such differences was not measured.
- Seed 42 is the earlier run of each version, made before `--seed` existed. `v3b` was not
  replicated across seeds.
- The correctness verdicts for the two `v3a` seed-42 models are not in the repository. Their
  inference files predate the `question` field the correctness judge reads.
- Each response was judged once per judge. Judge noise was not measured beyond the 64 validation
  labels.
- `beta` and the other DPO hyperparameters are still the untuned defaults of Week 6.

## How to run

Everything from the repository root. Training and inference need the GPU; judging needs
`OPENAI_API_KEY`.

```bash
# SFT and DPO for one data version and seed
python -m grounded_refusal.train.train_sft --data data/data_v3c_train.jsonl --seed 44
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> --seed 44

# Held-out inference (--adapter of a DPO checkpoint merges its SFT adapter itself)
python -m grounded_refusal.inference.run_inference --data data/data_v3b_heldout.jsonl \
  --adapter checkpoints/<run> \
  --output outputs/inference-qwen2.5-3b-instruct-dpo/<run>_v3b_heldout.jsonl

# Both judges; rerun with --resume if it stops
python -m grounded_refusal.eval.run_eval \
  --input outputs/inference-qwen2.5-3b-instruct-dpo/<run>_v3b_heldout.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct-dpo/<run>_v3b_heldout_eval_judge-gpt5-mini.jsonl \
  --correctness-output outputs/eval-qwen2.5-3b-instruct-dpo/<run>_v3b_heldout_correctness_judge-gpt5-mini.jsonl
```

## Next

Open questions, not decisions:

- Why the DPO stage produces the extra remark is unexplained. The ablation removes one candidate
  and leaves the question open.
- The conditions under which the remark is false are known from six rows. Paired test items that
  hold the fact and the question fixed and vary only the evidence (with or without an unasked
  detail, consistent or inconsistent with common knowledge) would test them on more than six.
- The false remark is a claim about what the evidence lacks, the same kind of claim a refusal
  makes. Scoring such claims directly would replace the manual count with a metric.
- Issue #14, a per-tag breakdown of held-out metrics computed in code, is still open, as is the
  templated-answer problem of issue #19.
