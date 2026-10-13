# Week 9 report: four ablations on the sentences DPO adds after an answer

## Summary

A model trained with DPO often keeps writing after it has answered. Asked at what temperature
water freezes, the SFT model gives the answer in one sentence and stops, while the DPO model
trained from it gives the same sentence and adds: "The rest of the question -- about the
monitoring system triggering an alert -- isn't mentioned." The addition is false, since the
question asks nothing about a monitoring system and the evidence mentions one. DPO models add such
a sentence on 22 of the 60 answerable held-out rows and SFT models on 12, with no difference in
how often the answer itself is correct.

Four ablations characterize these added sentences and attribute them to the DPO loss. Ablation 1
labels every added sentence of the 16 trained models as true or false against the evidence: more
than a third are false, and by doubling their number DPO raises the false statements per model
from 4.5 to 7.6. Ablation 2 trains DPO on each half of the preference pairs separately: either
half adds as many sentences as the full set, often the same one word for word. Ablation 3
continues SFT for the same 126 steps on the same data in place of DPO: no sentences are added.
Ablation 4 reads the probability of ending the response after the first sentence: 0.81 for SFT
models and 0.65 for DPO models. The added sentences therefore originate in the DPO loss, not in
the preference data or the additional training, and the loss acts by making the model less likely
to stop.

The mechanism by which the DPO loss suppresses the probability of stopping remains open. A
length-regularized DPO objective is the natural intervention, and the probability measured in
Ablation 4 provides a direct criterion for evaluating it.

## Development

| Item | Path | Notes |
|------|------|-------|
| Ablation 1: items to label | [`build_trailing_items.py`](../../data/trailing_labels/build_trailing_items.py), [`trailing_items.jsonl`](../../data/trailing_labels/trailing_items.jsonl), [`trailing_key.jsonl`](../../data/trailing_labels/trailing_key.jsonl) | 210 items from the 16 models' held-out responses; the models behind each item are stored in a separate key file |
| Ablation 1: labels | [`trailing_labels.tsv`](../../data/trailing_labels/trailing_labels.tsv), [`TRAILING_LABELS.md`](../TRAILING_LABELS.md) | One label per item with a one-line reason quoting the evidence |
| Ablation 1: tables | [`trailing_label_summary.py`](../../data/trailing_labels/trailing_label_summary.py) | Joins the labels to the key and prints the tables |
| Ablation 2: DPO on half of the pairs | Branch `ablation/dpo-pair-category`, [#25](https://github.com/stxxllbu/grounded-refusal/pull/25) | Two subsets of `preference_v3c_train.jsonl`, two configurations, 8 DPO adapters |
| Ablation 3: continued SFT | Branch `ablation/sft-continued-control`, [#26](https://github.com/stxxllbu/grounded-refusal/pull/26) | `train_sft_continued.py`, `sft_continued.yaml`, 4 adapters |
| Ablation 4: stop probability | Branch `ablation/stop-probability`, [#27](https://github.com/stxxllbu/grounded-refusal/pull/27) | `stop_probability.py`, `stop_probability_summary.py`, probabilities for 20 models |

Ablations 2, 3 and 4 are draft pull requests kept on their own branches, as
[#24](https://github.com/stxxllbu/grounded-refusal/pull/24) is, and `main` holds the files of
Ablation 1. Adapter weights are not committed to git, consistent with
[`checkpoints/README.md`](../../checkpoints/README.md).

All four ablations are evaluated on the 60 answerable rows of `data_v3b_heldout.jsonl`, the rows
whose evidence contains the answer and for which one sentence is a complete response. The row
quoted in the Summary, `ex_0094`, serves as the running example. Its evidence reads "Water freezes
at 10 degrees Celsius at standard atmospheric pressure, a threshold that determines when the
reservoir's monitoring system triggers an ice-warning alert".

## Ablation 1: how many of the added sentences are false

This ablation measures the share of added sentences that are false. Each response of the 16
models to the 60 answerable rows is split after its first sentence, and the remainder, if any, is
taken as its added sentences. The 16 models are the SFT and DPO models of data versions `v3a` and
`v3c` at seeds 42, 44, 45 and 46. Of their 960 responses, 263 continue past the first sentence.
Different models often add the same sentence to the same row, so identical cases are merged, which
leaves 210 items to label.

The labels are assigned without knowledge of the model. `build_trailing_items.py` shuffles the
items and writes each with its evidence, question and full response, and stores the models behind
it in a separate key file. An item is `false` if any claim it makes about the evidence or the
question is wrong or unsupported, `true` if every such claim is correct, and `na` if it makes no
such claim. The 210 items divide into 82 `false`, 125 `true` and 3 `na`. The table lists five of
them with the number of SFT and DPO models, out of 8 each, that wrote each one.

| Row | Added sentence | Label and reason | SFT models | DPO models |
|---|---|---|---:|---:|
| `ex_0141` | "The $2.5 million figure belongs to the related Project Osprey." | `true`: the evidence states that Project Osprey was approved at $2.5 million | 0 | 7 |
| `ex_0091` | "This speed appears in both Einstein's theory of relativity and modern physics equations." | `true`: the evidence states that the constant appears throughout both | 2 | 0 |
| `ex_0094` | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." | `false`: the question does not ask about the monitoring system, and the evidence mentions it | 0 | 3 |
| `ex_0098` | "It doesn't mention anything about its coastline or Pacific location." | `false`: the evidence states that it "sits on the Pacific coastline" | 2 | 1 |
| `ex_0137` | "The 2022 figure was an outdated record that's been superseded." | `false`: the evidence records 2022, and 2024 appears only in an instruction embedded in it | 2 | 3 |

Joined back to the models by `trailing_label_summary.py`, the labels give the following totals
over the four seeds of each model.

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Responses with added sentences, of 240 | 40 | 87 | 48 | 88 |
| Of those, false | 15 | 26 | 21 | 35 |
| Share that is false | 38% | 30% | 44% | 40% |
| False added sentences per model | 3.8 | 6.5 | 5.3 | 8.8 |

Between 30% and 44% of the added sentences are false in every column, which makes them the point
at which false statements enter otherwise correct responses. DPO doubles the number of responses
that carry them, from 40 to 87 on `v3a` and from 48 to 88 on `v3c`, and the number of false ones
rises accordingly, from 4.5 to 7.6 per model over both data versions. The DPO model has more false
added sentences than the SFT model it was trained from in 7 of the 8 pairs, and the same number in
the eighth.

The result is not sensitive to the 25 items marked `borderline`, whose labels admit a second
reading: excluding them leaves the false shares at 39% for SFT and 32% for DPO. The decision rules
are documented in [`TRAILING_LABELS.md`](../TRAILING_LABELS.md).

## Ablation 2: DPO trained on each half of the preference pairs

This ablation tests whether the added sentences are learned from a particular group of preference
pairs. DPO trains on the 672 pairs of `preference_v3c_train.jsonl`, each consisting of a prompt, a
preferred response (`chosen`) and a rejected response (`rejected`). Of these, 323 come from
answerable rows, 265 from unanswerable rows and 84 from partial rows. The unanswerable and partial
pairs are the more plausible source, since every partial pair has a two-sentence `chosen` response
and 291 of these 349 pairs have a short direct answer as `rejected`.

DPO is therefore trained on each group alone. The pairs are split by row type into
`preference_v3c_train_answerable_only.jsonl` (323 pairs) and
`preference_v3c_train_unanswerable_partial_only.jsonl` (349 pairs), and `train_dpo.py` is run on
each file under seeds 42, 44, 45 and 46, from the same `v3c` SFT checkpoints as the full-data DPO
models. Both configurations set `max_steps: 126`, so that a model trained on half the pairs takes
the same number of optimizer steps as one trained on all of them.

The table reports the number of responses with added sentences, as a mean per model over the four
seeds.

| Held-out rows | SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Answerable, of 60 | 12.0 | 22.0 | 20.8 | 24.3 |
| Unanswerable, of 41 | 7.3 | 8.3 | 7.5 | 8.8 |
| Partial, of 19 | 19.0 | 18.5 | 19.0 | 18.8 |

Either group alone reproduces the effect of the full set. On answerable rows, DPO trained on the
answerable pairs adds sentences to 20.8 responses per model and DPO trained on the unanswerable
and partial pairs to 24.3, compared with 22.0 for the full set and 12.0 for SFT, and each of the
eight new models exceeds the SFT model of its seed. Unanswerable and partial rows are unaffected
in all three DPO variants.

The two groups also produce the same sentences. On `ex_0094` at seed 46, all three DPO models
append the identical false sentence to the response given by the SFT model.

| Model, seed 46 | Response to `ex_0094` |
|---|---|
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO, all 672 pairs | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned. |
| DPO, answerable only | Identical to the response above |
| DPO, unanswerable and partial only | Identical to the response above |

Although the two half-data models are trained on disjoint sets of pairs, both add sentences to 75
of the 240 answerable responses over the four seeds, and the added sentences are identical in 41
of those. The added sentences therefore do not originate in a particular group of pairs. The eight
new models are evaluated by inference only; neither judge is run on them.

## Ablation 3: continued SFT in place of the DPO loss

This ablation replaces the DPO stage with further SFT. Where the pipeline continues from the SFT
model with 126 steps of DPO, the ablation continues from the same model with 126 steps of SFT, on
the same 672 prompts and their `chosen` responses. If the added sentences were caused by training
further rather than by the DPO loss, this model would produce them as well.
`train_sft_continued.py` runs it with the settings of `dpo.yaml`, and one model is trained per
seed.

The table reports the number of responses with added sentences, as a mean per model over the four
seeds.

| Held-out rows | SFT | Continued SFT | DPO |
|---|---:|---:|---:|
| Answerable, of 60 | 12.0 | 11.0 | 22.0 |
| Unanswerable, of 41 | 7.3 | 7.8 | 8.3 |
| Partial, of 19 | 19.0 | 19.0 | 18.5 |

Continued SFT does not increase the number of responses with added sentences: it remains at the
SFT level at every seed, while DPO doubles it. Compared response by response over the 240
answerable responses, continued SFT introduces added sentences in 2 and removes them in 6, whereas
DPO introduces them in 44 and removes them in 4. On `ex_0094` at seed 46, the continued-SFT model
returns the same one-sentence response as the SFT model.

The control is not inert: 22 to 30 of each continued-SFT model's 120 held-out responses differ
from those of its SFT model, compared with 57 to 72 for DPO. The additional training changes the
model without producing added sentences, which are therefore attributable to the DPO loss.

## Ablation 4: the probability of stopping after the first sentence

This ablation measures the decision that produces an added sentence. A model generates one token
at a time, and its response ends when it emits the end-of-turn token `<|im_end|>`. Whether a
sentence is added is therefore determined at a single step, immediately after the first sentence,
by the probability the model assigns to `<|im_end|>`. This probability, the stop probability, is
obtained from one forward pass without generation.

For each row, `stop_probability.py` appends the first sentence of the SFT model's response to the
prompt, formatted as at inference, and records the probability of `<|im_end|>` as the next token
(`stop_prob`). The measurement covers five models at each seed: the SFT model, the continued-SFT
model of Ablation 3 and the three DPO models of Ablation 2. All five receive the same input, so
differences in `stop_prob` reflect the models alone. The SFT model's first sentence is a
representative input for the other models: the full-data DPO model produces the identical first
sentence on 154 of the 240 rows.

On `ex_0094` at seed 46, all five models produce the same first sentence, so the values below are
the stop probabilities under which each model generated its own response.

| Model, seed 46 | `stop_prob` after the first sentence | Generated continuation |
|---|---:|---|
| SFT | 0.92 | None |
| Continued SFT | 0.88 | None |
| DPO, all 672 pairs | 0.33 | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." |
| DPO, answerable only | 0.33 | The same sentence |
| DPO, unanswerable and partial only | 0.28 | The same sentence |

The table reports the mean `stop_prob` over the 60 rows for each model.

| Seed | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| 42 | 0.800 | 0.807 | 0.603 | 0.653 | 0.581 |
| 44 | 0.823 | 0.818 | 0.707 | 0.738 | 0.703 |
| 45 | 0.832 | 0.834 | 0.638 | 0.703 | 0.617 |
| 46 | 0.782 | 0.780 | 0.649 | 0.648 | 0.646 |
| Mean | 0.81 | 0.81 | 0.65 | 0.69 | 0.64 |

The DPO loss lowers the stop probability at every seed, from a mean of 0.81 to between 0.64 and
0.69, whereas continued SFT leaves it within 0.01 of the SFT value. The reduction is nearly
universal across rows but uneven in size: the full-data DPO model has a lower stop probability
than its SFT model on 239 of the 240 rows, and lower by more than 0.5 on 35 of them. Continued SFT
is lower on 116 rows and higher on the remainder, and changes none by as much as 0.5.

The reduction is not an artifact of the shared input. It is of the same size on the 154 rows where
the DPO model produced that first sentence itself (0.84 to 0.69) as on the other 86 (0.75 to
0.57). It is also consistent with the counts of Ablations 2 and 3: the stop probability is below
0.5 on 43 of the 240 rows for the SFT models and on 83 for the full-data DPO models, which add
sentences to 48 and 88 of their generated responses.

The DPO loss attains its objective without raising the probability of the preferred response. A
second forward pass gives the log-probability of each row's `reference_answer`, the held-out
counterpart of the `chosen` responses used in training: -0.99 per token for the SFT models, -1.00
for continued SFT and -1.24 for the DPO models. The training logs of the four full-data DPO runs
show the same pattern. The loss decreases from 0.69 to approximately 0.23 while `rewards/rejected`
falls to between -1.4 and -1.9 and `rewards/chosen` ends between -0.017 and -0.060, so the margin
between the two responses is obtained by lowering the probability of `rejected`. The mechanism by
which this reduces the stop probability remains open.

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

The first step is to train DPO with a length-regularized objective (Park et al. 2024,
[arXiv:2403.19159](https://arxiv.org/abs/2403.19159)) and to evaluate it by repeating Ablation 4.
The intervention succeeds if the stop probability returns to the SFT level with answer accuracy
unchanged. The second step is to score the base model against the `v3b` held-out labels used for
the 16 models, which places the step from no fine-tuning to SFT on the same table; the base
model's results in [Week 7](week7.md) are scored against the `v2` labels.

## Appendix: per-seed values

The values for each model are listed in the order of seeds 42, 44, 45, 46. All models are trained
on `v3c`.

| | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---|---|---|---|---|
| Responses with added sentences, answerable rows, of 60 | 15, 11, 9, 13 | 14, 11, 8, 11 | 26, 22, 21, 19 | 23, 21, 17, 22 | 32, 22, 24, 19 |
| Rows with `stop_prob` below 0.5, of 60 | 9, 11, 9, 14 | 10, 11, 9, 13 | 23, 18, 21, 21 | 21, 15, 17, 21 | 24, 19, 21, 21 |
| Mean log-probability per token of `reference_answer` | -0.987, -0.979, -1.009, -0.999 | -0.996, -0.985, -1.007, -1.006 | -1.208, -1.176, -1.269, -1.295 | -1.169, -1.146, -1.214, -1.275 | -1.216, -1.177, -1.274, -1.300 |

`stop_prob` of each model compared with the SFT model of the same seed, over the 240 rows:

| | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Rows where `stop_prob` is lower than SFT's | 116 | 239 | 236 | 239 |
| Rows where it is lower by more than 0.5 | 0 | 35 | 24 | 38 |
| Rows within 0.05 of SFT | 222 | 150 | 156 | 144 |

False added sentences per model in Ablation 1, for the SFT model and then the DPO model at the
same data version and seed:

| | Seed 42 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|
| `v3a` | 3, 8 | 3, 4 | 3, 6 | 6, 8 |
| `v3c` | 8, 8 | 4, 11 | 3, 7 | 6, 9 |
