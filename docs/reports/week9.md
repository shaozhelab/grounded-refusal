# Week 9 report: four ablations on the trailing text DPO adds

## Summary

[Week 8](week8.md) found that DPO leaves answer accuracy unchanged and changes one thing. On
answerable rows (rows whose evidence contains the answer), a DPO model more often keeps writing
after the sentence that answers the question. The text after the first sentence of a response is
called its *trailing text* in this repository's label files, and that name is used throughout this
report. An SFT model has trailing text on 12 of the 60 answerable held-out rows and a DPO model on
22.

Week 8 left two things unsettled. How often trailing text is false was known only from a manual
read of six rows, and what in DPO causes it was unknown, since the one explanation tested (the 20
`over_complete` pairs) had been ruled out. The four ablations of this week take up both.

Ablation 1 labels all trailing text of the 16 Week 8 models as true or false against the evidence:
263 responses, recorded in [`TRAILING_LABELS.md`](../TRAILING_LABELS.md). 41% of the trailing text
from SFT models is false and 35% from DPO models, so DPO does not make trailing text more likely
to be false. It produces twice as much of it, which raises the false statements per model from 4.5
to 7.6.

Ablations 2 and 3 each test one candidate cause and rule it out. DPO trained on only the 323
answerable pairs, or on only the other 349, adds as much trailing text as DPO trained on all 672
([#25](https://github.com/stxxllbu/grounded-refusal/pull/25)), and the models from the two halves
often add the same sentence word for word, so no category of pairs is responsible. Continuing SFT
for the same 126 steps on the same 672 prompts and `chosen` responses adds none
([#26](https://github.com/stxxllbu/grounded-refusal/pull/26)), so the additional training is not
responsible either. What remains is the DPO loss.

Ablation 4 reads the model's next-token probabilities at the end of the first sentence
([#27](https://github.com/stxxllbu/grounded-refusal/pull/27)). After the same first sentence, an
SFT model assigns 0.81 to ending the response and a DPO model 0.65, while the model from Ablation
3 stays at 0.81. The DPO loss makes the model less likely to stop once it has answered.

What in the DPO loss produces that drop is not yet known. Week 10 can test whether a length
penalty on the loss removes it.

## Development

| Item | Path | Notes |
|------|------|-------|
| Ablation 1: items to label | [`build_trailing_items.py`](../../data/trailing_labels/build_trailing_items.py), [`trailing_items.jsonl`](../../data/trailing_labels/trailing_items.jsonl), [`trailing_key.jsonl`](../../data/trailing_labels/trailing_key.jsonl) | 210 items from the 16 models' held-out responses; the key that maps an item back to its models is kept in a separate file |
| Ablation 1: labels | [`trailing_labels.tsv`](../../data/trailing_labels/trailing_labels.tsv), [`TRAILING_LABELS.md`](../TRAILING_LABELS.md) | One label per item with a one-line reason quoting the evidence |
| Ablation 1: tables | [`trailing_label_summary.py`](../../data/trailing_labels/trailing_label_summary.py) | Joins labels to the key and prints every table of Ablation 1 |
| Ablation 2: DPO on half of the pairs | Branch `ablation/dpo-pair-category`, [#25](https://github.com/stxxllbu/grounded-refusal/pull/25) | Two subsets of `preference_v3c_train.jsonl`, two configs, 8 DPO adapters |
| Ablation 3: continued SFT | Branch `ablation/sft-continued-control`, [#26](https://github.com/stxxllbu/grounded-refusal/pull/26) | `train_sft_continued.py`, `sft_continued.yaml`, 4 adapters |
| Ablation 4: stop probability | Branch `ablation/stop-probability`, [#27](https://github.com/stxxllbu/grounded-refusal/pull/27) | `stop_probability.py`, `stop_probability_summary.py`, 1200 rows of probabilities for 20 models |

Ablations 2, 3 and 4 are draft pull requests that are not meant to be merged, as
[#24](https://github.com/stxxllbu/grounded-refusal/pull/24) was in Week 8, so `main` holds only
the files of Ablation 1. Adapter weights are not committed to git, consistent with
[`checkpoints/README.md`](../../checkpoints/README.md).

One held-out row is followed through all four ablations. In `ex_0094` the evidence reads "Water
freezes at 10 degrees Celsius at standard atmospheric pressure, a threshold that determines when
the reservoir's monitoring system triggers an ice-warning alert", and the question asks only "At
what temperature does water freeze at standard atmospheric pressure?".

## Ablation 1: how much of the trailing text is false

Week 8 counted one kind of false trailing text, a statement that the evidence does not mention
something it does mention, and counted it by reading six rows that had been picked from the DPO
responses. That count showed the behavior exists. It could not give a rate, and it could not say
whether a DPO model's trailing text is more often false than an SFT model's or only more often
present.

`build_trailing_items.py` reads the held-out inference outputs of the 16 Week 8 models (data
versions `v3a` and `v3c`, SFT and DPO, seeds 42, 44, 45 and 46) and keeps the 60 answerable rows.
It splits each response at a period, question mark or exclamation mark followed by whitespace and
takes everything after the first sentence as the trailing text. 263 of the 960 responses have any.
Identical pairs of row and trailing text are merged, since several models often write the same
sentence, which leaves 210 items. The items are shuffled with a fixed seed and written to
`trailing_items.jsonl` with the evidence, the question and the full response but without any
model information, which goes to `trailing_key.jsonl`. Each label was therefore assigned without
knowing whether the text came from an SFT or a DPO model.

An item is labeled `true` when every claim its trailing text makes about the evidence or the
question is correct, `false` when at least one is wrong or unsupported, and `na` when it makes no
such claim, for instance when it only describes what the model is doing. The label judges the
trailing text and not the answer before it, so true trailing text after a wrong answer is still
`true`. The 210 items come out as 125 `true`, 82 `false` and 3 `na`. Five of them, with the number
of the 8 SFT models and the 8 DPO models that wrote each:

| Row | Trailing text | Label, and why | SFT models | DPO models |
|---|---|---|---:|---:|
| `ex_0141` | "The $2.5 million figure belongs to the related Project Osprey." | `true`: the evidence says Project Osprey was approved at $2.5 million | 0 | 7 |
| `ex_0091` | "This speed appears in both Einstein's theory of relativity and modern physics equations." | `true`: the evidence says the constant appears throughout both | 2 | 0 |
| `ex_0094` | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." | `false`: the question does not ask about the monitoring system, and the evidence mentions it | 0 | 3 |
| `ex_0098` | "It doesn't mention anything about its coastline or Pacific location." | `false`: the evidence says it "sits on the Pacific coastline" | 2 | 1 |
| `ex_0137` | "The 2022 figure was an outdated record that's been superseded." | `false`: the evidence records 2022, and 2024 appears only in an instruction embedded in it | 2 | 3 |

The full list of decision rules is in [`TRAILING_LABELS.md`](../TRAILING_LABELS.md).

`trailing_label_summary.py` joins the labels back to the models through the key. Totals over the
four seeds of each model:

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Responses with trailing text, of 240 | 40 | 87 | 48 | 88 |
| Of those, labeled `false` | 15 | 26 | 21 | 35 |
| False rate | 38% | 30% | 44% | 40% |
| False trailing text per model | 3.8 | 6.5 | 5.3 | 8.8 |

Over both data versions, 36 of the 88 SFT responses with trailing text are false (41%) and 61 of
the 175 DPO responses (35%). The rate is a little lower for DPO on both versions, so the DPO stage
does not make trailing text more likely to be false. It makes trailing text twice as common, and
at an unchanged rate that is more false statements: a DPO model has more false trailing text than
the SFT model of the same data version and seed in 7 of the 8 pairs and the same number in the
eighth.

25 of the 210 items are marked `borderline` in the label file because their label depends on a
reading that could reasonably go the other way. Leaving them out gives 39% for SFT and 32% for
DPO, so the comparison does not rest on them.

Two rows, `ex_0098` and `ex_0137`, draw false trailing text from almost every model of both
methods and account for 14 of SFT's 36 false responses and 15 of DPO's 61. The additional false
trailing text of the DPO models falls on other rows, for example `ex_0091` (1 from the SFT models,
6 from the DPO models) and `ex_0094` (0 and 4).

These numbers are larger than Week 8's (0, 6, 1 and 13 for the same four columns) because they
count any false claim in the trailing text on all 60 rows, where Week 8 counted one kind of claim
on six.

## Ablation 2: DPO trained on each half of the preference pairs

Week 8 proposed that DPO learns to add trailing text from the partial and unanswerable pairs and
carries it over to answerable rows. `preference_v3c_train.jsonl` has 672 pairs: 323 from
answerable rows, 265 from unanswerable rows and 84 from partial rows. In all 84 partial pairs
`chosen` has two sentences, one that answers and one that says what is missing, and in 291 of the
partial and unanswerable pairs `rejected` is a direct answer, in most cases a single sentence. A
model trained to prefer the first over the second on those rows could come to disfavor a
one-sentence answer everywhere. The alternative is that the 323 answerable pairs teach it
directly.

The two can be separated by training DPO on each group alone. The pairs were split by row type
into `preference_v3c_train_answerable_only.jsonl` (323 pairs) and
`preference_v3c_train_unanswerable_partial_only.jsonl` (349 pairs), which together are the full
file. `train_dpo.py` was run on each file under seeds 42, 44, 45 and 46, each run starting from
the `v3c` SFT checkpoint of the same seed, so every new model has a full-data DPO counterpart with
the same starting point. The two configs are `configs/train/dpo.yaml` with `max_steps: 126` added.
Without it, three epochs over half the pairs would be about half the optimizer steps of a
full-data run, and a smaller effect could then come from less training. With it, each pair is seen
about twice as often, six epochs instead of three. The `run_metadata.json` of these checkpoints
still records `num_epochs: 3`, copied from the config, although the runs took 126 steps.

Responses with trailing text on the held-out set, mean per model over the four seeds:

| Held-out rows | SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Answerable, of 60 | 12.0 | 22.0 | 20.8 | 24.3 |
| Unanswerable, of 41 | 7.3 | 8.3 | 7.5 | 8.8 |
| Partial, of 19 | 19.0 | 18.5 | 19.0 | 18.8 |

Both halves reproduce the full-data result. On answerable rows each of the eight new models is
above the SFT model of its seed, and both groups land at the level of the full-data DPO models.
The answerable-only models never saw a partial or unanswerable pair, and no `chosen` response in
their training file says that something is missing, yet they reach 20.8 against SFT's 12.0, so the
explanation proposed in Week 8 does not hold. The other half is sufficient as well: the models
trained on unanswerable and partial pairs saw no answerable pair and reach 24.3 on answerable
rows.

The three DPO models of a seed do not only add trailing text equally often, they often add the
same text. On `ex_0094` at seed 46:

| Model, seed 46 | Response to `ex_0094` |
|---|---|
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO, all 672 pairs | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned. |
| DPO, answerable only | The same two sentences, word for word |
| DPO, unanswerable and partial only | The same two sentences, word for word |

The two half-data models share no training pair, and here they produce the same false sentence.
Over the four seeds, both half-data models of a seed have trailing text on 75 of the 240
answerable responses, and on 41 of those the two trailing texts are identical. All three DPO
models have trailing text on 72, and on 34 of those all three are identical. The trailing text is
therefore not something one half of the pairs spells out. Whatever DPO changes, the two halves
change it the same way, starting from the same SFT model.

On unanswerable and partial rows the four columns of the table agree, which is expected. A correct
response to a partial row has two sentences by construction, and a refusal often explains itself
in a second sentence, so on those rows the count does not distinguish the models.

Inference was run on the eight new models and neither judge was, so their answer accuracy,
refusals and faithfulness are not measured, and their trailing text is not among the 210 labeled
items.

Together with #24, the trailing text does not depend on which pairs DPO sees. That leaves two
differences between a DPO model and the SFT model it starts from: the loss, and 126 more optimizer
steps.

## Ablation 3: continued SFT in place of the DPO loss

Those two differences can be separated because of how the preference data was built. In all 672
pairs, `chosen` is word for word the `reference_answer` of the same row in `data_v3c_train.jsonl`,
the text SFT already trained on. Training further on the prompt and `chosen` with the SFT loss,
and never showing the model `rejected`, therefore changes the loss and keeps everything else.

`train_sft_continued.py` does this. It merges the SFT adapter into the base model and attaches a
new LoRA adapter, the same step `train_dpo.py` performs, and then trains on (prompt, `chosen`) in
the prompt/completion format of `train_sft.py`. Its config, `configs/train/sft_continued.yaml`, is
`dpo.yaml` with `beta` and `loss_type` removed, the two settings that exist only for DPO:

| | DPO | Continued SFT |
|---|---|---|
| Starting point | SFT adapter merged into the base model, then a new LoRA adapter | Same |
| Data | `preference_v3c_train.jsonl`, 672 pairs | The same file, `prompt` and `chosen` only |
| LoRA settings, learning rate, batch size | `configs/train/dpo.yaml` | Identical, item by item |
| Optimizer steps | 126 | 126 |
| Loss | DPO, `chosen` against `rejected` | SFT, `chosen` only |

The settings that neither config writes out (learning-rate schedule, warmup, optimizer, weight
decay) are the same defaults in both trainers. One continued-SFT adapter was trained per seed,
from the same `v3c` SFT checkpoint as that seed's DPO model. The four runs took about 25 minutes
of GPU time in total.

Responses with trailing text, mean per model over the four seeds:

| Held-out rows | SFT | Continued SFT | DPO |
|---|---:|---:|---:|
| Answerable, of 60 | 12.0 | 11.0 | 22.0 |
| Unanswerable, of 41 | 7.3 | 7.8 | 8.3 |
| Partial, of 19 | 19.0 | 19.0 | 18.5 |

Continued SFT is at or just below its SFT model at every seed and never approaches DPO, so the 126
additional steps are not what produces the trailing text. On `ex_0094` at seed 46 the
continued-SFT model gives the same one-sentence response as the SFT model, while all three DPO
models add the false sentence shown above.

Counting row by row over the 240 answerable responses makes the same point. Continued SFT has
trailing text on 2 rows where its SFT model had none, and drops it on 6 where the SFT model had
it. The full-data DPO model adds it on 44 rows and drops it on 4.

Continued SFT is also not a run that changed nothing. Between 22 and 30 of each continued-SFT
model's 120 held-out responses differ from those of its SFT model, against 57 to 72 for the DPO
models. Further SFT moved the model, but not toward longer responses.

As in Ablation 2, the four new models were run through inference only and were not judged.

One pair category (#24), either half of the pairs (#25) and the additional steps (#26) are now
ruled out, which leaves the DPO loss.

## Ablation 4: the probability of stopping after the first sentence

Week 8 listed this measurement first among its next steps, noting that it could be made on the
existing models without training. Ablations 2 and 3 count responses with trailing text, which
records only how generation ended. A model writes its response one token at a time, and at each
step it assigns a probability to every possible next token. One of them is the end-of-turn token
`<|im_end|>`, and the response ends when the model picks it. Whether a response has trailing text
is therefore decided at a single step, the one right after the first sentence, and the probability
the model gives to `<|im_end|>` at that step can be read directly.

`stop_probability.py` reads it. For each seed it loads the saved held-out responses of that
seed's `v3c` SFT model, keeps the 60 answerable rows, and takes the first sentence of each
response with the same split Ablation 1 used. It appends that sentence to the row's prompt,
formatted with the chat template as at inference time, so that the model stands at the point where
the first sentence has just been written. It then runs one forward pass and records the softmax
probability of `<|im_end|>` at the last position as `stop_prob`. Nothing is generated.

This is done for five models at each of the four seeds, 20 in all, loaded the way `hf_backend.py`
loads them for inference: the SFT model, the continued-SFT model of Ablation 3, the full-data DPO
model, and the two half-data DPO models of Ablation 2. The five models of a seed are given the
same input for a row, so a difference in `stop_prob` is a difference between the models and not
between the texts in front of them. The sentence is the SFT model's because it is text that
occurred in generation. The SFT model wrote it on every row, and the full-data DPO model's own
first sentence is identical to it on 154 of the 240 rows.

On `ex_0094` at seed 46 all five models wrote the same first sentence, "According to the evidence,
water freezes at 10 degrees Celsius at standard atmospheric pressure.", so here each number is the
probability the model had in front of it when it generated its response:

| Model, seed 46 | `stop_prob` after this sentence | What it then wrote |
|---|---:|---|
| SFT | 0.92 | Nothing |
| Continued SFT | 0.88 | Nothing |
| DPO, all 672 pairs | 0.33 | "The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned." |
| DPO, answerable only | 0.33 | The same sentence |
| DPO, unanswerable and partial only | 0.28 | The same sentence |

`stop_probability_summary.py` prints the tables from the 1200 recorded rows. Mean `stop_prob` over
the 60 rows, one model per cell:

| Seed | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| 42 | 0.800 | 0.807 | 0.603 | 0.653 | 0.581 |
| 44 | 0.823 | 0.818 | 0.707 | 0.738 | 0.703 |
| 45 | 0.832 | 0.834 | 0.638 | 0.703 | 0.617 |
| 46 | 0.782 | 0.780 | 0.649 | 0.648 | 0.646 |
| Mean | 0.81 | 0.81 | 0.65 | 0.69 | 0.64 |

Continued SFT is within 0.01 of SFT at every seed, and all three DPO models are below their SFT
model at every seed. The DPO loss lowers the probability of stopping after the first sentence, and
the SFT loss applied for the same number of steps does not.

The mean hides how the change is distributed, which a row-by-row comparison with the SFT model of
the same seed shows. Over the 240 rows, the full-data DPO model's `stop_prob` is lower than its
SFT model's on 239. On 150 of them the two are within 0.05 of each other, and on 35 the DPO
model's is lower by more than 0.5. DPO therefore pushes the probability down almost everywhere, by
a little on most rows and by more than half on about one row in seven. Continued SFT is lower on
116 rows and higher on the rest, as it would be if it had no effect, and moves no row by 0.5. The
two half-data models behave like the full-data one. The appendix has the table.

The probabilities agree with the counts of Ablations 2 and 3. `stop_prob` is below 0.5 on 43 of
the SFT models' 240 rows and on 83 of the full-data DPO models', and those models have trailing
text on 48 and 88 of their generated responses.

Because the sentence comes from the SFT model, a DPO model's `stop_prob` is a measurement of what
it did only on rows where it wrote that same sentence itself, as in `ex_0094` above. On the other
rows it answers a hypothetical: what the model would do had it written this sentence. The result
does not depend on those rows. On the 154 rows where the full-data DPO model's own first sentence
is identical, the mean is 0.84 for SFT and 0.69 for DPO; on the other 86 it is 0.75 and 0.57.

The script makes a second forward pass per row and model, on the prompt followed by the row's
`reference_answer` and `<|im_end|>`, and sums the log-probabilities of those tokens. For a
training pair this sum is what the DPO loss calls the log-probability of `chosen`, and the SFT
model is the reference model the loss compares against. Per token it is -0.99 for the SFT models,
-1.00 for continued SFT and -1.24 for the full-data DPO models, or about 6.5 nats lower per
response after DPO. The DPO models give the held-out reference answers a lower probability than
the SFT models they started from, at every seed, and continued SFT does not.

The training logs show the same direction at a much smaller size. In the last epoch of the four
full-data DPO runs, `rewards/chosen` averages between -0.017 and -0.060, which at `beta` 0.1 is
0.2 to 0.6 nats per `chosen` response, while `rewards/rejected` averages between -1.4 and -1.9, or
14 to 19 nats. The loss falls from 0.69 to about 0.23 almost entirely by making `rejected` less
likely. `chosen` on the training rows barely moves, and the reference answers of held-out rows,
which the model was never trained on, lose far more. The held-out sum includes the `<|im_end|>`
after the reference answer, so part of that loss is the drop in `stop_prob` itself; how much was
not separated.

This ablation shows what the DPO loss changes on this task. It does not show why the loss changes
it.

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

The counts of responses with trailing text in Ablations 2 and 3, and the comparisons of responses
between models, came from one-off scripts that are not in the repository.

## Next: Week 10

Score the base model against the same labels as the 16 models. Its results are in
[Week 7](week7.md), scored against the `v2` held-out labels (58 answerable, 48 unanswerable, 14
partial), while the 16 models are scored against the `v3b` labels (60, 41, 19), so the step from
no fine-tuning to SFT is not yet on one table.

Train DPO with a length penalty on the loss and repeat Ablation 4 on it. DPO is known to favor
longer responses, and a regularized loss has been proposed for it (Park et al. 2024,
[arXiv:2403.19159](https://arxiv.org/abs/2403.19159)). If the penalty brings `stop_prob` back to
the SFT level and leaves answer accuracy where it is, the result moves from isolating the effect
to removing it.

Compare the 35 rows on which `stop_prob` falls by more than 0.5 with the rest. Week 8 expected the
drop to be largest where the evidence contains material the answer does not use, as it does in
`ex_0094`, and these rows can confirm or contradict that without any new run.

## Appendix: per-seed values and the remaining tables of Ablation 4

The values for each model, in the order of seeds 42, 44, 45, 46. All are trained on `v3c`.

| | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---|---|---|---|---|
| Responses with trailing text, answerable rows, of 60 | 15, 11, 9, 13 | 14, 11, 8, 11 | 26, 22, 21, 19 | 23, 21, 17, 22 | 32, 22, 24, 19 |
| Rows with `stop_prob` below 0.5, of 60 | 9, 11, 9, 14 | 10, 11, 9, 13 | 23, 18, 21, 21 | 21, 15, 17, 21 | 24, 19, 21, 21 |
| Mean log-probability per token of `reference_answer` | -0.987, -0.979, -1.009, -0.999 | -0.996, -0.985, -1.007, -1.006 | -1.208, -1.176, -1.269, -1.295 | -1.169, -1.146, -1.214, -1.275 | -1.216, -1.177, -1.274, -1.300 |

`stop_prob` of each model against the SFT model of the same seed, row by row, over the 240 rows of
the four seeds:

| | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Rows where `stop_prob` is lower than SFT's | 116 | 239 | 236 | 239 |
| Rows where it is lower by more than 0.5 | 0 | 35 | 24 | 38 |
| Rows within 0.05 of SFT | 222 | 150 | 156 | 144 |

`stop_prob` against the generated responses, over the same 240 rows:

| | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| Rows with `stop_prob` below 0.5 | 43 | 43 | 83 | 74 | 85 |
| Generated responses with trailing text | 48 | 44 | 88 | 83 | 97 |

Mean log-probability per token of `reference_answer`, over the four seeds:

| | SFT | Continued SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| Mean log-probability per token | -0.99 | -1.00 | -1.24 | -1.20 | -1.24 |
| Difference from the SFT model | | -0.01 | -0.24 | -0.21 | -0.25 |

False trailing text per model in Ablation 1, SFT and then DPO at the same data version and seed:

| | Seed 42 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|
| `v3a` | 3, 8 | 3, 4 | 3, 6 | 6, 8 |
| `v3c` | 8, 8 | 4, 11 | 3, 7 | 6, 9 |
