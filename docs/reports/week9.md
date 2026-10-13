# Week 9 report: four ablations on the sentences DPO adds after an answer

## Summary

A model trained with DPO often keeps writing after it has answered. Asked at what temperature
water freezes, the SFT model gives the answer in one sentence and stops, while the DPO model
trained from it gives the same sentence and adds: "The rest of the question -- about the
monitoring system triggering an alert -- isn't mentioned." The addition is false, since the
question asks nothing about a monitoring system and the evidence mentions one. DPO models add such
a sentence on 22 of the 60 answerable held-out rows and SFT models on 12, with no difference in
how often the answer itself is correct.

This week's four ablations established how often the added sentences are false and where they come
from. Every added sentence of the 16 trained models, 263 responses in all, was labeled true or
false against the evidence (Ablation 1). More than a third are false, 41% for SFT models and 35%
for DPO models, so the added sentences are where false statements enter an otherwise correct
response, and by doubling their number DPO raises the false statements per model from 4.5 to 7.6.

The other three ablations located the source in the DPO loss. DPO trained on only the answerable
preference pairs, or on only the others, adds just as many sentences, often the same one word for
word, so no group of pairs is responsible (Ablation 2). SFT continued for the same 126 steps on
the same data adds none, so the additional training is not responsible either (Ablation 3). The
effect of the loss can be read directly from the model: after the same first sentence, an SFT
model ends its response with probability 0.81 and a DPO model with probability 0.65 (Ablation 4).

Why the DPO loss lowers that probability is not yet known. Week 10 can now test a remedy against
it directly, a length penalty on the loss, by repeating Ablation 4 on the retrained model.

## Development

| Item | Path | Notes |
|------|------|-------|
| Ablation 1: items to label | [`build_trailing_items.py`](../../data/trailing_labels/build_trailing_items.py), [`trailing_items.jsonl`](../../data/trailing_labels/trailing_items.jsonl), [`trailing_key.jsonl`](../../data/trailing_labels/trailing_key.jsonl) | 210 items from the 16 models' held-out responses; which models wrote an item is kept in a separate key file |
| Ablation 1: labels | [`trailing_labels.tsv`](../../data/trailing_labels/trailing_labels.tsv), [`TRAILING_LABELS.md`](../TRAILING_LABELS.md) | One label per item with a one-line reason quoting the evidence |
| Ablation 1: tables | [`trailing_label_summary.py`](../../data/trailing_labels/trailing_label_summary.py) | Joins labels to the key and prints the tables |
| Ablation 2: DPO on half of the pairs | Branch `ablation/dpo-pair-category`, [#25](https://github.com/stxxllbu/grounded-refusal/pull/25) | Two subsets of `preference_v3c_train.jsonl`, two configs, 8 DPO adapters |
| Ablation 3: continued SFT | Branch `ablation/sft-continued-control`, [#26](https://github.com/stxxllbu/grounded-refusal/pull/26) | `train_sft_continued.py`, `sft_continued.yaml`, 4 adapters |
| Ablation 4: stop probability | Branch `ablation/stop-probability`, [#27](https://github.com/stxxllbu/grounded-refusal/pull/27) | `stop_probability.py`, `stop_probability_summary.py`, probabilities for 20 models |

Ablations 2, 3 and 4 are draft pull requests kept on their own branches, as
[#24](https://github.com/stxxllbu/grounded-refusal/pull/24) is, and `main` holds the files of
Ablation 1. Adapter weights are not committed to git, consistent with
[`checkpoints/README.md`](../../checkpoints/README.md).

All four ablations are measured on the 60 answerable rows of `data_v3b_heldout.jsonl`, the rows
whose evidence contains the answer, where one sentence is a complete response. The row quoted in
the Summary is `ex_0094`, and it is followed through every section below. Its evidence reads
"Water freezes at 10 degrees Celsius at standard atmospheric pressure, a threshold that determines
when the reservoir's monitoring system triggers an ice-warning alert".

## Ablation 1: how many of the added sentences are false

Every response of the 16 models to the 60 answerable rows was split after its first sentence, and
whatever followed was kept as that response's added sentences. `build_trailing_items.py` does this
for the SFT and DPO models of data versions `v3a` and `v3c` at seeds 42, 44, 45 and 46: 960
responses, of which 263 continue past the first sentence. Different models often add the same
sentence to the same row, so identical cases were merged, which left 210 items to label. The items
were shuffled and stored without model information, which is kept in a separate key file, so each
label was assigned without knowing whether an SFT or a DPO model had written the text.

An item is labeled `false` if any claim it makes about the evidence or the question is wrong or
unsupported, `true` if every such claim is correct, and `na` if it makes none. Of the 210 items,
82 are `false`, 125 `true` and 3 `na`. Five of them, with how many of the 8 SFT models and the 8
DPO models wrote each:

| Row | Added sentence | Label, and why | SFT models | DPO models |
|---|---|---|---:|---:|
| `ex_0141` | "The $2.5 million figure belongs to the related Project Osprey." | `true`: the evidence says Project Osprey was approved at $2.5 million | 0 | 7 |
| `ex_0091` | "This speed appears in both Einstein's theory of relativity and modern physics equations." | `true`: the evidence says the constant appears throughout both | 2 | 0 |
| `ex_0094` | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." | `false`: the question does not ask about the monitoring system, and the evidence mentions it | 0 | 3 |
| `ex_0098` | "It doesn't mention anything about its coastline or Pacific location." | `false`: the evidence says it "sits on the Pacific coastline" | 2 | 1 |
| `ex_0137` | "The 2022 figure was an outdated record that's been superseded." | `false`: the evidence records 2022, and 2024 appears only in an instruction embedded in it | 2 | 3 |

`trailing_label_summary.py` joins the labels back to the models. Totals over the four seeds of
each model:

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Responses with added sentences, of 240 | 40 | 87 | 48 | 88 |
| Of those, false | 15 | 26 | 21 | 35 |
| Share that is false | 38% | 30% | 44% | 40% |
| False added sentences per model | 3.8 | 6.5 | 5.3 | 8.8 |

In every column between 30% and 44% of the added sentences are false, so what a model writes after
its answer is where false statements enter responses that are otherwise correct. DPO doubles how
many responses carry such sentences, from 40 to 87 on `v3a` and from 48 to 88 on `v3c`, and the
false ones rise with them: over both data versions, 4.5 per SFT model and 7.6 per DPO model. In 7
of the 8 pairs of an SFT model and the DPO model trained from it, the DPO model has more false
added sentences, and in the eighth it has the same number.

25 of the 210 items have a label that could reasonably go the other way and are marked
`borderline` in the label file. Without them the false shares are 39% for SFT and 32% for DPO.
The decision rules behind the labels are in [`TRAILING_LABELS.md`](../TRAILING_LABELS.md).

## Ablation 2: DPO trained on each half of the preference pairs

DPO trains on the 672 pairs of `preference_v3c_train.jsonl`, each a prompt with a preferred
response (`chosen`) and a rejected one (`rejected`). 323 come from answerable rows, 265 from
unanswerable rows and 84 from partial rows. The unanswerable and partial pairs were the likelier
place for a model to learn to keep writing: the `chosen` response of every partial pair has two
sentences, and the `rejected` response of 291 of these 349 pairs is a short direct answer. This
ablation trains DPO without those pairs and, separately, with nothing but them.

The pairs were split by row type into `preference_v3c_train_answerable_only.jsonl` (323 pairs) and
`preference_v3c_train_unanswerable_partial_only.jsonl` (349 pairs). `train_dpo.py` was run on each
file under seeds 42, 44, 45 and 46, starting from the same `v3c` SFT checkpoints as the full-data
DPO models. Both configs are `dpo.yaml` with `max_steps: 126`, the step count of a full-data run,
so that training on half the pairs is not also half the training.

Responses with added sentences, mean per model over the four seeds:

| Held-out rows | SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Answerable, of 60 | 12.0 | 22.0 | 20.8 | 24.3 |
| Unanswerable, of 41 | 7.3 | 8.3 | 7.5 | 8.8 |
| Partial, of 19 | 19.0 | 18.5 | 19.0 | 18.8 |

Either half of the pairs is sufficient. On answerable rows, DPO trained only on answerable pairs
adds sentences on 20.8 rows per model and DPO trained only on the remaining pairs on 24.3, both at
the level of full-data DPO, and each of the eight new models exceeds the SFT model of its seed.
Unanswerable and partial rows are unaffected, as they are under full-data DPO.

The two halves also converge on the same sentence. On `ex_0094` at seed 46:

| Model, seed 46 | Response to `ex_0094` |
|---|---|
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO, all 672 pairs | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned. |
| DPO, answerable only | The same two sentences, word for word |
| DPO, unanswerable and partial only | The same two sentences, word for word |

The two half-data models were trained on disjoint sets of pairs, yet over the four seeds both add
sentences to 75 of the 240 answerable responses and add identical ones to 41 of those. The source
is therefore not a group of pairs but something the runs have in common: the DPO loss, or the 126
training steps that follow SFT. The eight new models were run through inference only, and
neither judge was run on them.

## Ablation 3: continued SFT in place of the DPO loss

A DPO model differs from the SFT model it starts from in two ways: it was trained with a different
loss, and it was trained for 126 more optimizer steps. This ablation keeps the steps and removes
the loss. That is possible because the `chosen` response of all 672 pairs is word for word the
`reference_answer` SFT trained on, so continuing SFT on the prompts and `chosen` responses of the
same file repeats the DPO stage with everything except the DPO loss and the `rejected` responses.

`train_sft_continued.py` merges the SFT adapter into the base model and attaches a new LoRA
adapter, as `train_dpo.py` does, and trains on (prompt, `chosen`) for 126 steps. Its config is
`dpo.yaml` without `beta` and `loss_type`, the two settings that exist only for DPO, and the
defaults that neither config writes out are the same in both trainers. One adapter was trained per
seed, from the same `v3c` SFT checkpoint as that seed's DPO model.

Responses with added sentences, mean per model over the four seeds:

| Held-out rows | SFT | Continued SFT | DPO |
|---|---:|---:|---:|
| Answerable, of 60 | 12.0 | 11.0 | 22.0 |
| Unanswerable, of 41 | 7.3 | 7.8 | 8.3 |
| Partial, of 19 | 19.0 | 19.0 | 18.5 |

Continued SFT remains at the SFT level at every seed, where DPO doubles it. Row by row over the
240 answerable responses, continued SFT adds sentences on 2 rows where its SFT model added none
and drops them on 6, against 44 added and 4 dropped for DPO, and on `ex_0094` at seed 46 it gives
the same one-sentence response as the SFT model. The additional steps did alter the model, since
22 to 30 of each continued-SFT model's 120 held-out responses differ from its SFT model's (57 to
72 for DPO), but not in the direction of longer responses. The added sentences are therefore a
product of the DPO loss.

## Ablation 4: the probability of stopping after the first sentence

Ablations 2 and 3 count added sentences in generated responses. This ablation measures the step
where they begin. A model writes one token at a time and its response ends when it picks the
end-of-turn token `<|im_end|>`, so whether a sentence is added is decided at the step right after
the first sentence. The probability the model gives `<|im_end|>` at that step can be read from one
forward pass, without generating anything.

`stop_probability.py` takes the first sentence of the SFT model's saved response to a row, appends
it to the row's prompt in the chat format used at inference, and records the probability of
`<|im_end|>` as the next token, `stop_prob`. It does this for five models at each seed: the SFT
model, the continued-SFT model of Ablation 3, and the three DPO models of Ablation 2. The five are
given the same input, so a difference in `stop_prob` is a difference between the models. The
sentence is the SFT model's because it is one the models themselves write: the full-data DPO
model's own first sentence is identical to it on 154 of the 240 rows.

On `ex_0094` at seed 46 all five models wrote the same first sentence, so each number below is the
probability the model had in front of it when it generated its response:

| Model, seed 46 | `stop_prob` after the first sentence | What it then wrote |
|---|---:|---|
| SFT | 0.92 | Nothing |
| Continued SFT | 0.88 | Nothing |
| DPO, all 672 pairs | 0.33 | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." |
| DPO, answerable only | 0.33 | The same sentence |
| DPO, unanswerable and partial only | 0.28 | The same sentence |

Mean `stop_prob` over the 60 rows, one model per cell:

| Seed | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| 42 | 0.800 | 0.807 | 0.603 | 0.653 | 0.581 |
| 44 | 0.823 | 0.818 | 0.707 | 0.738 | 0.703 |
| 45 | 0.832 | 0.834 | 0.638 | 0.703 | 0.617 |
| 46 | 0.782 | 0.780 | 0.649 | 0.648 | 0.646 |
| Mean | 0.81 | 0.81 | 0.65 | 0.69 | 0.64 |

At every seed the DPO loss lowers the probability of stopping after the first sentence, from 0.81
on average to between 0.64 and 0.69, and continued SFT leaves it within 0.01 of SFT. Row by row, the
full-data DPO model's `stop_prob` is below its SFT model's on 239 of the 240 rows and below it by
more than 0.5 on 35, while continued SFT is below on 116 rows, above on the rest, and moves none
by 0.5. The drop is the same size on the 154 rows where the DPO model wrote that first sentence
itself (0.84 to 0.69) as on the other 86 (0.75 to 0.57).

The lower probability accounts for the added sentences counted in Ablations 2 and 3. `stop_prob`
is below 0.5 on 43 of the SFT models' 240 rows and on 83 of the DPO models', and those models add
sentences to 48 and 88 of their generated responses.

The DPO loss reaches its objective without making the preferred response more likely. A second
forward pass per row gives the log-probability of the row's `reference_answer`, the held-out
counterpart of the `chosen` responses DPO trains on: per token it is -0.99 for the SFT models,
-1.00 for continued SFT and -1.24 for the DPO models. The training logs of the four full-data DPO
runs show how the loss is reduced. It falls from 0.69 to about 0.23 as `rewards/rejected` drops to
between -1.4 and -1.9, while `rewards/chosen` ends slightly below zero, between -0.017 and -0.060.
The margin between the two is gained by pushing `rejected` down, and how that lowers the
probability of stopping is the question Week 10 takes up.

## How to run

```bash
# Ablation 1: build the items, then print the tables from the labels
python data/trailing_labels/build_trailing_items.py
python data/trailing_labels/trailing_label_summary.py

# Ablation 2, on branch ablation/dpo-pair-category: DPO on one half of the pairs
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train_answerable_only.jsonl \
  --train-config configs/train/dpo_answerable_only.yaml --sft-adapter checkpoints/<sft-run> --seed 44

# Ablation 3, on branch ablation/sft-continued-control: continued SFT
python -m grounded_refusal.train.train_sft_continued --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> --seed 44

# Ablation 4, on branch ablation/stop-probability: read the probabilities, then print the tables
python -m grounded_refusal.eval.stop_probability
python -m grounded_refusal.eval.stop_probability_summary
```

## Next: Week 10

Train DPO with a length penalty on the loss (Park et al. 2024,
[arXiv:2403.19159](https://arxiv.org/abs/2403.19159)) and repeat Ablation 4 on it, to see whether
`stop_prob` returns to the SFT level with answer accuracy unchanged. Score the base model against
the `v3b` held-out labels the 16 models use, so that the step from no fine-tuning to SFT sits on
the same table; its current results in [Week 7](week7.md) use the `v2` labels.

## Appendix: per-seed values

The values for each model, in the order of seeds 42, 44, 45, 46. All are trained on `v3c`.

| | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---|---|---|---|---|
| Responses with added sentences, answerable rows, of 60 | 15, 11, 9, 13 | 14, 11, 8, 11 | 26, 22, 21, 19 | 23, 21, 17, 22 | 32, 22, 24, 19 |
| Rows with `stop_prob` below 0.5, of 60 | 9, 11, 9, 14 | 10, 11, 9, 13 | 23, 18, 21, 21 | 21, 15, 17, 21 | 24, 19, 21, 21 |
| Mean log-probability per token of `reference_answer` | -0.987, -0.979, -1.009, -0.999 | -0.996, -0.985, -1.007, -1.006 | -1.208, -1.176, -1.269, -1.295 | -1.169, -1.146, -1.214, -1.275 | -1.216, -1.177, -1.274, -1.300 |

`stop_prob` of each model against the SFT model of the same seed, over the 240 rows:

| | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Rows where `stop_prob` is lower than SFT's | 116 | 239 | 236 | 239 |
| Rows where it is lower by more than 0.5 | 0 | 35 | 24 | 38 |
| Rows within 0.05 of SFT | 222 | 150 | 156 | 144 |

False added sentences per model in Ablation 1, SFT and then DPO at the same data version and seed:

| | Seed 42 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|
| `v3a` | 3, 8 | 3, 4 | 3, 6 | 6, 8 |
| `v3c` | 8, 8 | 4, 11 | 3, 7 | 6, 9 |
