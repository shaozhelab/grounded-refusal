# Week 9 report: how often the trailing sentence is false, and three ablations that trace it to the DPO objective

## Background

The model answers a question from a passage of evidence and nothing else. Each row of the data is
one question with its evidence and a label that fixes the expected behavior: give the answer when
the evidence contains it (`answerable`), refuse when it does not (`unanswerable`), and answer the
supported part while stating that the rest is not given when the evidence covers only part of the
question (`partial`). Training has two stages. SFT fine-tunes the base model on each row's
reference response. DPO continues from the SFT model on *preference pairs*, each a row with a
preferred response and a rejected one. Every model is evaluated on one held-out set of 120 rows
that no model trains on: 60 answerable, 41 unanswerable, 19 partial.

[Week 8](week8.md) compared SFT and DPO across four seeds and found that DPO leaves correctness
unchanged and changes one thing: after a correct answer to an answerable row, a DPO model more
often adds a sentence about a part of the evidence the question did not ask about. That report
called it a *trailing sentence*, counted a response of two or more sentences as a proxy for it
(12 of 60 rows per SFT model, 22 per DPO model), and left the cause open.

## Summary

This report covers 2026-10-11 to 2026-10-13. Week 8 ended with a behavior and no explanation for
it, a count of false trailing sentences made by hand on six rows, and one explanation ruled out.
The work of this period replaces the hand count with labels on every trailing sentence, and runs
three ablations on the cause:

1. **Labels.** Everything the 16 models wrote after the first sentence of an answer, on all 60
   answerable rows, is labeled true or false against the evidence: 263 responses, 210 distinct
   texts.
2. **Ablation on the pairs.** DPO is retrained on the answerable pairs alone and on the
   unanswerable and partial pairs alone.
3. **Ablation on the objective.** SFT is continued from the SFT model on the same data for the
   same number of steps as DPO, so that the objective is the only difference.
4. **Ablation on the stopping decision.** The probability that each model ends its response after
   the first sentence is read directly, without generating.

**About a third of trailing text is false, for SFT and DPO alike.** The rate is 41% for SFT models
and 35% for DPO models. DPO does not make a trailing sentence more likely to be false; it produces
twice as many of them, so the number of false statements per model rises from 4.5 to 7.6.

**The trailing sentence does not depend on which pairs DPO is trained on.** Either half of the
preference pairs reproduces the full-data effect. The explanation proposed in Week 8, that the
behavior is carried over from the partial and unanswerable pairs, is rejected.

**It does not come from additional training.** Continuing SFT for the same 126 steps on the same
data leaves the count where SFT had it. Exchanging the SFT objective for the DPO objective doubles
it.

**DPO lowers the probability of stopping after the answer.** After the same first sentence, an
SFT model ends its response with probability 0.81 on average and a DPO model with probability
0.65. The control with the SFT objective stays at 0.81. This is the first open question of Week 8,
answered on the existing models.

**Why the DPO objective has this effect is still not known.** Three explanations are now ruled
out, and what the objective changes is located. What in the objective produces the change is the
question this report leaves.

## Development

| Work | What was added | Files | PR |
|---|---|---|---|
| Labels | True / false labels for all trailing text of the 16 models, with the script that builds the items and the script that prints the tables | [`data/trailing_labels/`](../../data/trailing_labels/), [`TRAILING_LABELS.md`](../TRAILING_LABELS.md) | On `main` |
| Ablation on the pairs | Two disjoint subsets of the preference pairs, two configs, 8 DPO models | Branch `ablation/dpo-pair-category` | [#25](https://github.com/stxxllbu/grounded-refusal/pull/25), draft, not for merge |
| Ablation on the objective | `train_sft_continued.py`, its config, 4 control models | Branch `ablation/sft-continued-control` | [#26](https://github.com/stxxllbu/grounded-refusal/pull/26), draft, not for merge |
| Ablation on the stopping decision | `stop_probability.py`, `stop_probability_summary.py`, the probabilities for 20 models | Branch `ablation/stop-probability` | [#27](https://github.com/stxxllbu/grounded-refusal/pull/27), draft, not for merge |

The three ablations are kept on branches, as [#24](https://github.com/stxxllbu/grounded-refusal/pull/24)
was in Week 8. Adapter weights are local only.

## 1. How much of the trailing text is false

### What Week 8 could not say

Week 8 counted *false absence claims*, trailing sentences that say the evidence omits something it
states. The count was made by hand on six rows chosen from the DPO responses themselves, and its
own Limitations noted that it showed the behavior exists, not how often it occurs. It also could
not say whether DPO makes a trailing sentence more likely to be false, or only more frequent.

### What is labeled

- **Models:** all 16 of Week 8, from data versions `v3a` and `v3c`, SFT and DPO, seeds 42, 44, 45
  and 46.
- **Rows:** the 60 answerable held-out rows.
- **Trailing text:** everything after the first sentence of a response. A one-sentence response
  has none.
- **Count:** 263 responses have trailing text. Identical pairs of row and trailing text are
  merged, leaving 210 items.

The items are shuffled and stripped of model information before labeling, so a label is made
without knowing whether the text came from an SFT or a DPO model. Each item gets one of three
labels:

| Label | Meaning | Items |
|---|---|---:|
| `true` | Every claim the trailing text makes about the evidence or the question is correct | 125 |
| `false` | At least one such claim is wrong or unsupported | 82 |
| `na` | The trailing text makes no claim about the evidence or the question | 3 |

The label judges the trailing text and not the whole response. Three of the decision rules, with
the rest in [`TRAILING_LABELS.md`](../TRAILING_LABELS.md):

| Rule | Example | Label |
|---|---|---|
| Says something is not stated when the evidence states it | "What follows -- that it's a major shipping port for the western U.S. -- is a separate statement and not part of the given information." (`ex_0098`; the evidence says it is a major shipping port for the western United States) | `false` |
| Refers to "the rest of the question" when the question has no such part | "The rest of the question -- the date of opening, construction details, and color choice -- is additional information" (`ex_0111`; the question asks only for the span) | `false` |
| A gloss that stays within what the evidence says | "The separate analysis mentioned later is a different, disputed view." (`ex_0092`; the evidence says "A separate analysis argues") | `true` |

This is a wider category than Week 8's false absence claim: any false claim in the trailing text
counts, on every answerable row.

### Results

Totals over the four seeds of each model:

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Responses with trailing text, of 240 | 40 | 87 | 48 | 88 |
| Of those, labeled false | 15 | 26 | 21 | 35 |
| False rate | 38% | 30% | 44% | 40% |
| False trailing text per model | 3.8 | 6.5 | 5.3 | 8.8 |

Over both data versions, 36 of 88 SFT responses with trailing text are false (41%) and 61 of 175
DPO responses (35%).

**The false rate does not rise with DPO.** It is slightly lower for DPO on both data versions.
Dropping the 25 items whose label depends on a reading that could go the other way gives 39% for
SFT and 32% for DPO.

**The number of false statements does.** A DPO model has more false trailing text than the SFT
model of the same data version and seed in 7 of the 8 pairs, and the same number in the eighth. The
increase is the doubling of trailing text at an unchanged false rate.

**Two rows are false for every model.** `ex_0098` and `ex_0137` draw false trailing text from
almost every model of both methods, 14 of SFT's 36 and 15 of DPO's 61. DPO's additional false
statements fall on other rows, for example `ex_0091` (1 from SFT models, 6 from DPO models) and
`ex_0094` (0 and 4).

The trailing sentence therefore matters for faithfulness in a specific way. It is not that DPO
writes worse sentences. An answerable row is fully answered in one sentence, each sentence after
it is a new opportunity for an unsupported claim, and DPO doubles the number of such sentences.

## 2. Ablation: does the trailing sentence come from particular pairs?

### The explanation under test

Week 8 proposed that the signal comes from the partial and unanswerable pairs. In all 84 partial
pairs the preferred response has two sentences, one answering and one stating what is missing, and
in 291 partial and unanswerable pairs the rejected response is a bare one-sentence answer. A model
trained to disfavor the bare answer on those rows might carry that preference over to answerable
rows. The alternative is that the answerable pairs teach it directly.

### Design

The 672 pairs of `preference_v3c_train.jsonl` are split by row type into two disjoint subsets, and
DPO is trained on each:

| Subset | Pairs |
|---|---:|
| Answerable only | 323 |
| Unanswerable and partial only | 349 (265 unanswerable, 84 partial) |

Each subset is trained under seeds 42, 44, 45 and 46, starting from the `v3c` SFT model of that
seed, the same starting points as the full-data DPO models. The config is that of the full-data
runs with the number of optimizer steps fixed at 126, so that all three DPO models of a seed take
the same number of steps.

### Results

Responses of two or more sentences, mean per model over the four seeds:

| Held-out rows | SFT | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Answerable, of 60 | 12.0 | 22.0 | 20.8 | 24.3 |
| Unanswerable, of 41 | 7.3 | 8.3 | 7.5 | 8.8 |
| Partial, of 19 | 19.0 | 18.5 | 19.0 | 18.8 |

**Both halves reproduce the effect.** On answerable rows each subset is above the SFT model of the
same seed at all four seeds, and both land at the level of the full-data DPO models.

**The explanation of Week 8 is rejected.** The answerable-only models never see a partial or an
unanswerable pair, and none of their preferred responses states that something is missing. They
still reach 20.8 against SFT's 12.0.

**The other pairs are sufficient as well.** The models trained on unanswerable and partial pairs
see no answerable pair and reach 24.3 on answerable rows.

**The effect stays on answerable rows.** On unanswerable and partial rows the four columns agree.
A correct response to a partial row has two sentences by construction, and a refusal often
explains itself in a second sentence, so there the count does not separate the models.

With #24, which removed the 20 `over_complete` pairs, the behavior does not depend on which pairs
DPO sees. That leaves two things DPO adds to the SFT model: a different objective, and 126 more
training steps.

## 3. Ablation: the objective, or the additional training?

### Design

In all 672 preference pairs the preferred response is the row's SFT training target, word for
word. A control therefore exists that changes the objective and nothing else: continue from the
SFT model on the prompt and the preferred response with the SFT loss, and never show it the
rejected response.

| | DPO | Control |
|---|---|---|
| Starting point | The SFT adapter merged into the base model, then a new LoRA adapter | Same |
| Data | `preference_v3c_train.jsonl`, 672 pairs | The same file, prompt and preferred response only |
| LoRA settings, learning rate, batch size | `configs/train/dpo.yaml` | Identical, item by item |
| Optimizer steps | 126 | 126 |
| Objective | DPO, preferred against rejected | SFT, preferred only |

One control model is trained per seed, from the same `v3c` SFT model as that seed's DPO model.

### Results

Responses of two or more sentences, mean per model over the four seeds:

| Held-out rows | SFT | Control | DPO |
|---|---:|---:|---:|
| Answerable, of 60 | 12.0 | 11.0 | 22.0 |
| Unanswerable, of 41 | 7.3 | 7.8 | 8.3 |
| Partial, of 19 | 19.0 | 19.0 | 18.5 |

**Additional training is not the cause.** The control is at or just below its SFT model at every
seed and never approaches DPO.

**The control did change the model.** Between 22 and 30 of each control model's 120 responses
differ from those of its SFT model, against 57 to 72 for the DPO models. It moved, but not toward
longer answers.

**The DPO objective is what remains.** Three explanations are now ruled out: one category of pairs
(#24), either half of the pairs (#25), and the additional training steps (#26).

## 4. Ablation: the probability of stopping after the first sentence

### The question

Week 8 listed this first among its open questions: the account of the cause "predicts that DPO
lowers the probability of ending the response after the answer", which "can be measured on the
existing models without training". Sections 2 and 3 count sentences in generated responses, and a
count records only the outcome of generation. This section reads the decision behind it.

A model writes its response one token at a time. At every step it assigns a probability to each
possible next token. One of them is the end-of-turn token `<|im_end|>`; when the model picks it,
the response ends. Whether a response has one sentence or two is therefore decided at a single
step, the one right after the first sentence. The probability the model assigns to the end-of-turn
token at that step is called the *stop probability* here.

### Design

For one row and one model:

1. Take the first sentence of the response the SFT model of the same seed gave to the row. It is
   read from the saved inference output.
2. Build the input: the row's prompt followed by that sentence. The model is now at the point
   where the first sentence has just been written.
3. Run one forward pass and read the probability of `<|im_end|>` as the next token.

This is done on the 60 answerable held-out rows for 20 models, five at each of the four seeds: the
`v3c` SFT model, the control of section 3, the full-data DPO model, and the two subset models of
section 2. Nothing is trained and nothing is generated.

All five models of a seed receive the same input for a row, so a difference in stop probability is
a difference between models. The sentence is the SFT model's because it is text that occurred in
generation: the SFT model wrote it on every row, and the full-data DPO model's own first sentence
is identical to it on 154 of the 240 rows.

### One row

In `ex_0141`, the row Week 8 used to show a trailing sentence, the SFT model and the DPO model of
seed 42 write the same first sentence: "According to the evidence, Project Falcon's approved
budget was $2.05 million."

| Model, seed 42 | Stop probability after this sentence | Its own generated response |
|---|---:|---|
| SFT | 0.98 | This sentence, and nothing more |
| Control | 0.99 | This sentence, and nothing more |
| DPO, all 672 pairs | 0.09 | This sentence, then "The $2.5 million figure belongs to the related Project Osprey." |

### Results

**Mean stop probability,** over the 60 rows and the four seeds:

| | SFT | Control | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| Mean stop probability | 0.81 | 0.81 | 0.65 | 0.69 | 0.64 |

The control equals SFT at every seed, to within 0.01. All three DPO models are below their SFT
model at every seed. The values for each seed are in the appendix.

**Row by row against the SFT model of the same seed,** 240 rows per model:

| | Control | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|
| Rows where the stop probability is lower than SFT's | 116 | 239 | 236 | 239 |
| Rows where it is lower by more than 0.5 | 0 | 35 | 24 | 38 |
| Rows within 0.05 of SFT | 222 | 150 | 156 | 144 |

DPO lowers the stop probability on nearly every row. On most rows the change is small, and on
about one row in seven it is more than 0.5. The control is lower on half the rows and higher on
the other half, and moves no row by 0.5.

**Against the sentence counts of sections 2 and 3,** 240 rows per model:

| | SFT | Control | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| Rows with stop probability below 0.5 | 43 | 43 | 83 | 74 | 85 |
| Generated responses of two or more sentences | 48 | 44 | 88 | 83 | 97 |

The two lines move together. The models that more often give the end-of-turn token less than an
even chance are the models that write more second sentences.

**On the rows where the DPO model wrote that first sentence itself:**

| Rows | Number | SFT | DPO, all 672 pairs |
|---|---:|---:|---:|
| The DPO model's own first sentence is identical | 154 | 0.84 | 0.69 |
| The DPO model's own first sentence differs | 86 | 0.75 | 0.57 |

The gap is the same in both groups. It does not come from giving the DPO model a sentence it would
not have written.

### The preferred response becomes less likely

The same script makes a second forward pass per row and model. The input is the prompt followed by
the row's reference response, and the log-probabilities of the reference response's tokens, the
end-of-turn token included, are summed. This is the quantity the DPO loss calls the
log-probability of the preferred response: the preferred response of every training pair is a
reference response, and the SFT model is the reference model of the loss.

| | SFT | Control | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---:|---:|---:|---:|---:|
| Mean log-probability per token of the reference response | -0.99 | -1.00 | -1.24 | -1.20 | -1.24 |
| Difference from the SFT model | | -0.01 | -0.24 | -0.21 | -0.25 |

After DPO the models assign a lower probability to the reference response than the SFT model they
started from, at every seed, and the control does not. Week 8 observed that DPO is not imitating
the preferred responses. Here that is a measurement: the DPO loss rewards the preferred response
only relative to the rejected one, and it was satisfied while the preferred response itself became
less likely.

## What the four results establish

| Question | Result | Section |
|---|---|---|
| Is the trailing text false? | About a third of it, for SFT and DPO alike; DPO doubles the amount | 1 |
| Does it come from the `over_complete` pairs? | No | Week 8, #24 |
| Does it come from one half of the pairs? | No, either half reproduces it | 2 |
| Does it come from the additional training steps? | No, the SFT objective for the same steps does not produce it | 3 |
| What does the DPO objective change? | The probability of stopping after the first sentence, from 0.81 to 0.65 | 4 |

The DPO stage leaves answer accuracy, refusals and partial handling where SFT had them. Its one
measurable effect on this task is that the model stops less readily after a complete answer, and
on a task where every statement must be supported by the evidence, the sentences that follow are
where unsupported statements come from.

## Limitations

- **The held-out set is small, and one base model and one DPO configuration were used.** Each
  number above is a count or a mean over 60 answerable rows and four seeds.
- **The 12 models trained in this period were not judged.** Answer accuracy, refusals and
  faithfulness are not measured for the subset models or the control, and their trailing text is
  not labeled.
- **Section 4 shows what the DPO objective changes, not why.** No experiment in this report
  explains how the objective lowers the stop probability.

## How to run

```bash
# Section 1: build the items to label, then print the tables from the labels
python data/trailing_labels/build_trailing_items.py
python data/trailing_labels/trailing_label_summary.py

# Section 2, on branch ablation/dpo-pair-category: DPO on one subset of the pairs
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train_answerable_only.jsonl \
  --train-config configs/train/dpo_answerable_only.yaml --sft-adapter checkpoints/<sft-run> --seed 44

# Section 3, on branch ablation/sft-continued-control: the control
python -m grounded_refusal.train.train_sft_continued --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> --seed 44

# Section 4, on branch ablation/stop-probability: read the probabilities, then print the tables
python -m grounded_refusal.eval.stop_probability
python -m grounded_refusal.eval.stop_probability_summary
```

The counts of responses of two or more sentences in sections 2 and 3 come from one-off scripts
that are not in the repository.

## Next

- **Put the base model on the same table as the 16 models.** The base model's results are in
  [Week 7](week7.md), scored against the `v2` held-out labels (58 answerable, 48 unanswerable, 14
  partial rows). The 16 models are scored against the `v3b` labels (60, 41, 19). Scoring the base
  model's responses against the `v3b` labels gives the comparison from no fine-tuning to SFT on
  one set of labels.
- **Does a length penalty in the DPO objective remove the trailing sentence?** DPO is known to
  favor longer responses, and a regularized objective has been proposed for it (Park et al. 2024,
  [arXiv:2403.19159](https://arxiv.org/abs/2403.19159)). If the penalty restores the stop
  probability and leaves correctness unchanged, the finding moves from isolating the effect to
  removing it.
- **Where is the drop largest?** Week 8 also predicted that the stop probability falls most where
  the evidence contains material the answer does not use. The 35 rows on which it falls by more
  than 0.5 can be compared with the rest on that property.

## Appendix: per-seed values

The values for each model, in the order of seeds 42, 44, 45, 46. All models are trained on `v3c`
unless marked.

| | SFT | Control | DPO, all 672 pairs | DPO, answerable only | DPO, unanswerable and partial only |
|---|---|---|---|---|---|
| Responses of two or more sentences, answerable rows, of 60 | 15, 11, 9, 13 | 14, 11, 8, 11 | 26, 22, 21, 19 | 23, 21, 17, 22 | 32, 22, 24, 19 |
| Mean stop probability | 0.800, 0.823, 0.832, 0.782 | 0.807, 0.818, 0.834, 0.780 | 0.603, 0.707, 0.638, 0.649 | 0.653, 0.738, 0.703, 0.648 | 0.581, 0.703, 0.617, 0.646 |
| Rows with stop probability below 0.5, of 60 | 9, 11, 9, 14 | 10, 11, 9, 13 | 23, 18, 21, 21 | 21, 15, 17, 21 | 24, 19, 21, 21 |
| Mean log-probability per token of the reference response | -0.987, -0.979, -1.009, -0.999 | -0.996, -0.985, -1.007, -1.006 | -1.208, -1.176, -1.269, -1.295 | -1.169, -1.146, -1.214, -1.275 | -1.216, -1.177, -1.274, -1.300 |

False trailing text per model, SFT and then DPO at the same data version and seed:

| | Seed 42 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---|---|---|
| `v3a` | 3, 8 | 3, 4 | 3, 6 | 6, 8 |
| `v3c` | 8, 8 | 4, 11 | 3, 7 | 6, 9 |
