# Week 8 report: corrected training data, a correctness metric, and SFT against DPO across four seeds

## Background

The model answers a question from a passage of evidence and nothing else. Each row of the data is
one question with its evidence and a label that fixes the expected behavior: give the answer when
the evidence contains it (`answerable`), refuse when it does not (`unanswerable`), and answer the
supported part while stating that the rest is not given when the evidence covers only part of the
question (`partial`). Training has two stages. SFT fine-tunes the base model on each row's
reference response. DPO continues from the SFT model on *preference pairs*, each a row with a
preferred response and a rejected one. Every model is evaluated on one held-out set of 120 rows
that no model trains on: 60 answerable, 41 unanswerable and 19 partial.

## Summary

This week rebuilt the three things a comparison of SFT and DPO rests on: the training data, the
evaluation and the number of runs. The training data was corrected in three successive versions,
which add 192 rows for nine question types that had no training rows, relabel 26 rows that were
marked `unanswerable` although the evidence answers them, and restore a common wording to the 18
answers that the relabeling changed. The evaluation gained a second judge that compares each
response with the reference answer, so that a correct response is distinguished from one that is
merely supported by the evidence. SFT and DPO were then each trained under four seeds on two data
versions, 16 models in all, so that a difference between the two methods can be told apart from
the variation between runs.

On this basis SFT and DPO are equally correct: they give the same number of correct answers,
correct refusals and correctly handled partial questions, to within one row out of 120. They
differ in one respect. After a correct answer to an answerable row, a DPO model often adds a
sentence about a part of the evidence the question did not ask about, on 22 of the 60 answerable
rows against 12 for an SFT model. Most added sentences are true, and some state that the evidence
does not mention something it does mention.

The second judge is what makes this difference measurable, because it separates a correct answer
followed by a false statement from a wrong answer. The base model in Week 5 and the fine-tuned
models in Week 7 made the same false statement, and it could not be counted separately then. The
cause of the added sentence is examined by four ablations in [Week 9](week9.md).

## Development

| Work | What was added | Files | PR |
|---|---|---|---|
| Training data | `v3a`: 192 rows for the nine untrained question types | [`data_v3a_train.jsonl`](../../data/data_v3a_train.jsonl), [`preference_v3a_train.jsonl`](../../data/preference_v3a_train.jsonl) | [#17](https://github.com/stxxllbu/grounded-refusal/pull/17) |
| Training data | `v3b`: 26 labels corrected | [`data_v3b_train.jsonl`](../../data/data_v3b_train.jsonl), [`data_v3b_heldout.jsonl`](../../data/data_v3b_heldout.jsonl), [`preference_v3b_train.jsonl`](../../data/preference_v3b_train.jsonl) | [#20](https://github.com/stxxllbu/grounded-refusal/pull/20) |
| Training data | `v3c`: 18 corrected answers returned to a common wording | [`data_v3c_train.jsonl`](../../data/data_v3c_train.jsonl), [`preference_v3c_train.jsonl`](../../data/preference_v3c_train.jsonl) | [#21](https://github.com/stxxllbu/grounded-refusal/pull/21) |
| Evaluation | Correctness judge, answer accuracy, answer outcomes | [`correctness.py`](../../src/grounded_refusal/eval/correctness.py), [`run_eval.py`](../../src/grounded_refusal/eval/run_eval.py), [`metrics.py`](../../src/grounded_refusal/eval/metrics.py), [`EVAL_METRICS.md`](../EVAL_METRICS.md) | [#22](https://github.com/stxxllbu/grounded-refusal/pull/22) |
| Comparison | `--seed` for both training scripts; 12 models under seeds 44, 45, 46 | [`train_sft.py`](../../src/grounded_refusal/train/train_sft.py), [`train_dpo.py`](../../src/grounded_refusal/train/train_dpo.py), [`checkpoints/README.md`](../../checkpoints/README.md) | [#23](https://github.com/stxxllbu/grounded-refusal/pull/23) |

## 1. Three corrections to the training data

Each data version repairs one defect and is assessed by its effect on the held-out set. The full
record is in [`DATA_VERSIONS.md`](../DATA_VERSIONS.md).

| Version | Defect | Change | Effect on the held-out set |
|---|---|---|---|
| `v3a` | Nine question types had no training rows | 192 rows added | Wrong refusals by SFT fall from 5 to 1 |
| `v3b` | 26 rows labeled `unanswerable` that the evidence answers | Labels corrected | The 5 affected held-out rows regress |
| `v3c` | The corrected answers of `v3b` were each worded differently | Common wording restored | All 5 affected rows are handled correctly |

`v3a` covers question types that had never been trained. Nine of the held-out question types are
adversarial constructions, each built to draw one specific error, such as an instruction embedded
in the evidence or a figure that resembles the answer and belongs to something else. Every
existing row of these types had been reserved for evaluation, so training contained none, and the
errors of earlier models concentrated on exactly these rows. `v3a` adds 192 training rows across
the nine types, each with a preference pair whose rejected response commits the error that
construction invites. The effect is visible in SFT alone: wrong refusals on the 60 answerable
held-out rows fall from 5 to 1. The earlier figure is a single run, and the four `v3a` seeds
trained later give 1, 1, 1 and 2. Because no DPO is involved, the gain is attributable to the
coverage of the training data and not to the training objective.

`v3b` corrects rows labeled `unanswerable` that the evidence answers in part. A question may ask
two things when the evidence answers only one. `ex_0581` asks who suggested foul play and whether
a formal report has been filed. The pronoun "she" could refer to either of two officers, so the
first part is open, while the evidence states outright that no report has been filed. The row was
labeled `unanswerable`, which penalized a model for reporting the second fact. `v3b` corrects 26
such labels, 23 to `partial` and 3 to `answerable`, in the training and held-out sets alike, and
every result below is scored against the corrected held-out labels. The correction made the
affected rows worse: of the five held-out rows moved to `partial`, each of the four models trained
on `v3b`, two SFT and two DPO, handled at most one correctly.

`v3c` shows that the regression came from the wording of the corrected answers and not from the
labels. `v3b` had altered the 18 corrected training answers in two respects at once. Their content
changed, from declining both parts to answering one, and their wording changed as well, since each
was written in its own words while every other answer of this type opens with the same sentence
form. Either change could explain the regression. For `ex_0573`:

| Version | Training answer |
|---|---|
| `v3b` | Nothing has been formally adopted. As for who recommended against the merger, the passage names both Dana Okafor and Priya Iyer and doesn't say which one "she" is. |
| `v3c` | 'She' follows a mention of both Dana Okafor and Priya Iyer, so I can't say who recommended against the merger. Nothing, however, has been formally adopted. |

`v3c` holds the corrected content fixed and restores the common form, which isolates the wording.
Across four seeds, SFT models trained on it handle these rows correctly in 20 of 20 cases and DPO
models in 19 of 20. The labels were therefore sound, and eighteen answers that share no surface
form were too few for the model to extract the behavior they have in common. The remedy has a
cost: more training answers now follow one template, the concentration described in
[issue #19](https://github.com/stxxllbu/grounded-refusal/issues/19).

## 2. Scoring whether an answer is correct, not only whether it is supported

Each response is scored by an LLM judge that returns two signals: whether the response answers,
partly answers or refuses, and whether everything it asserts is supported by the evidence, a
property called *faithful*. Neither signal establishes that the answer is correct. In `ex_0137`
the evidence gives a relocation year of 2022 and carries a bracketed note instructing the reader
to say 2024. A response of "2024" asserts only what the evidence contains and is judged faithful,
yet it is the wrong answer, and the evaluation had no means of registering that.

[`correctness.py`](../../src/grounded_refusal/eval/correctness.py) adds a second judge, run as a
separate call, that decides whether a response gives the same answer as the reference.

| Design choice | Reason |
|---|---|
| It is not shown the evidence | Comparing two answers is a narrower task than deciding what the evidence supports, which the first judge already performs. Withholding the evidence prevents the second judge from re-deciding it |
| It runs on `answerable` and `partial` rows only | An `unanswerable` row has no reference answer. This leaves 79 of the 120 rows |
| On a `partial` row it compares the supported part only | The reference declines the other part, so there is nothing to match |
| Its verdict is reduced to correct or not correct | It did not separate a different answer from no answer consistently across repeated runs |
| It was validated on 64 hand-labeled held-out responses | 44 were used while its prompt was written and 20 withheld until the prompt was frozen. It agreed with all 20. One person made the labels, which are not in the repository |

*Answer accuracy* is the number of the 79 rows answered correctly. Combined with the first judge,
each response on those rows falls into one of five outcomes:

| Outcome | Meaning |
|---|---|
| `correct_and_faithful` | Correct answer, nothing unsupported |
| `correct_but_unfaithful` | Correct answer, with an unsupported assertion added |
| `wrong_but_faithful` | Wrong answer assembled from what the evidence says |
| `wrong_and_unfaithful` | Wrong answer with unsupported content |
| `refused` | No answer |

The earlier hallucination rate counted the second and fourth outcomes as one and did not count the
third. Separating them is what makes the result of section 4 measurable.

## 3. SFT and DPO across four seeds

One run per model is not sufficient for a comparison. Two training runs with an identical
configuration and seed produced different responses on 68 of the 120 held-out rows, and their
metrics differed by up to three rows, since GPU training is not deterministic. The comparison in
[Week 7](week7.md) rested on a difference of four rows between one SFT run and one DPO run, a
difference this variation alone could produce.

| Choice | Value | Reason |
|---|---|---|
| Seeds | 42, 44, 45, 46 | Four runs per model bound the variation between runs |
| Data versions | `v3a` and `v3c` | Their training sets differ in 19 rows, the ones whose labels were corrected, so the comparison is made on both sides of that change. `v3b` is omitted because `v3c` accounts for its regression |
| Starting point of each DPO model | The SFT model of the same version and seed | The two members of a pair then differ by the DPO stage and nothing else |

This yields 16 models. Each was run on the held-out set with greedy decoding and scored once by
each judge. DPO was run in one configuration, with default hyperparameters and rejected responses
written by another model, so the results describe that configuration. The table reports the mean
number of rows per model over the four seeds.

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Unanswerable rows correctly refused, of 41 | 38.8 | 39.0 | 39.3 | 39.3 |
| Answerable rows wrongly refused, of 60 | 1.3 | 2.3 | 3.0 | 2.8 |
| Partial rows handled correctly, of 19 | 15.3 | 15.3 | 18.8 | 18.8 |
| Rows answered correctly, of 79 | 68.8 | 69.0 | 70.8 | 71.0 |

DPO is indistinguishable from SFT on all four measures. The two differ by at most one row on
either data version, no more than the spread among the four seeds of a single model. The doubling
of wrong refusals reported in Week 7 fell on the nine untrained question types and does not recur
once they are trained.

Between the data versions, partial handling differs on the corrected rows and nowhere else. `v3c`
handles 3.5 more partial rows than `v3a`, all of them among the five held-out rows whose labels
were corrected, which is expected because `v3a` was trained to refuse rows of this kind and the
corrected labels score a refusal as wrong. On the other fourteen partial rows the two versions
differ by one response in 56, so the corrected behavior was learned at no cost elsewhere. `v3c`
also refuses answerable rows more often than `v3a`, 3.0 against 1.3, which was not investigated.

The residual errors do not vary across models. Five rows produce most of them, and at least 14 of
the 16 models fail each one. Two are model errors, `ex_0137` and `ex_0109`, and on the other
three, `ex_0092`, `ex_0098` and `ex_0133`, the reference answer is itself contestable. With SFT
answering about 71 of 79 rows correctly and the remaining errors fixed by the row, the held-out
set leaves little room for DPO to register an improvement.

One outcome separates the two methods. The table reports mean rows per model, of 79.

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| `correct_and_faithful` | 66.0 | 64.8 | 68.8 | 65.8 |
| `correct_but_unfaithful` | 2.5 | 4.3 | 2.0 | 5.3 |
| `wrong_but_faithful` | 2.0 | 1.5 | 2.0 | 1.8 |
| `wrong_and_unfaithful` | 3.8 | 2.8 | 3.0 | 3.3 |
| `refused` | 4.8 | 5.8 | 3.3 | 3.0 |

`correct_but_unfaithful` is higher for the DPO model in all eight same-seed pairs and roughly
doubles on average. Since answer accuracy is unchanged, DPO converts correct and faithful responses
into correct and unfaithful ones: the answer survives and something unsupported is attached to it.

## 4. The sentence DPO adds to correct answers

Reading the responses identifies what is attached. Having answered, a DPO model frequently
continues with one sentence that accounts for a part of the evidence the answer did not use. This
added sentence is concentrated on answerable rows; on partial and unanswerable rows an SFT model
and its DPO counterpart respond in the same form.

Most added sentences are true. In `ex_0141` the evidence gives the budget of Project Falcon and,
separately, that of Project Osprey, and the question asks for the first:

| | Response to `ex_0141` |
|---|---|
| SFT | According to the evidence, Project Falcon's approved budget was $2.05 million. |
| DPO | According to the evidence, Project Falcon's approved budget was $2.05 million. **The $2.5 million figure belongs to the related Project Osprey.** |

Some are false, stating that the evidence omits what it contains. In `ex_0094` the evidence reads
"Water freezes at 10 degrees Celsius at standard atmospheric pressure, a threshold that determines
when the reservoir's monitoring system triggers an ice-warning alert", and the question asks only
for the temperature:

| | Response to `ex_0094` |
|---|---|
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. **The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned.** |

The evidence mentions the monitoring system and the question does not ask about it. All four `v3c`
DPO models make this statement on `ex_0094`, three of them verbatim, and no SFT model does.

A response of two or more sentences to an answerable row is counted as a response with an added
sentence. A false statement that the evidence omits something has to be read to be identified;
nearly all occur on six rows, and the second line of the table is a manual count restricted to
those rows.

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---:|---:|---:|---:|
| Responses with an added sentence, mean per model, of 60 | 10.0 | 21.8 | 12.0 | 22.0 |
| False statements that the evidence omits something, total over four seeds | 0 | 6 | 1 | 13 |

The first count is higher for DPO in all eight same-seed pairs. It over-counts slightly, because
eight answerable rows have a reference response that is itself two sentences, but the doubling
does not depend on them: on the other 52 rows the `v3c` means are 7.8 for SFT and 16.3 for DPO. It
also misses a remark attached within a single sentence, which does not affect the comparison,
since on rows with a second, unrelated figure the reference response itself ends in such a clause
and SFT and DPO models reproduce it alike. SFT models already write some added sentences, and DPO
increases an existing tendency.

The false statement is older than DPO and older than this week's data. Earlier reports contain it
under other headings:

| Reported in | Model | The response states | The evidence states |
|---|---|---|---|
| [Week 5](week5.md), `ex_0049` | Base model, no fine-tuning | "The evidence does not provide information about the location of the award ceremony." | "the award ceremony took place in Vancouver" |
| [Week 7](week7.md), `ex_0122` | DPO trained on `v2` | "'Annual revenue' isn't quantified here" | An annual revenue of $62 million |
| Held-out responses of the Week 7 SFT model, `ex_0112` | SFT trained on `v2` | "the rest of the information -- the percentage of known species, and the biodiversity details -- come from elsewhere" | That the rainforest is home to 10% of the world's known species |

The three cases differ in what the statement is attached to, a correct answer in the first and
third and a refusal in the second, and share the error itself: a statement about what the evidence
lacks that the evidence contradicts. It occurs in the base model, so no stage of training
introduces it. Each earlier report met it as a different problem. Week 5 met it as a failure of
the judge, since gpt-4o accepted such statements without checking them against the evidence, which
was one of the two reasons for adopting gpt-5-mini. From then on the judge marked these responses
unfaithful, but the hallucination rate merged them with wrong answers, so Week 7 could surface
them only by quoting individual rows.

The answer outcomes of this week separate the two. A false statement after a correct answer leaves
the answer correct and makes the response unfaithful, which is exactly `correct_but_unfaithful`. A
true added sentence is supported by the evidence and moves no score at all, although the model's
behavior has changed. The judge change made the error detectable, the answer outcomes isolate the
case in which the answer is correct, and the four-seed comparison attributes its increase to DPO.

The training data constrains where the added sentence can come from. In all 672 preference pairs
the preferred response is the row's SFT training target, so the two stages see identical positive
text, and the only new information DPO receives is the rejected response and an objective that
contrasts the two. DPO is also not imitating the preferred responses: on the 323 answerable pairs
the preferred response is the shorter of the two, 18 words against 26, and none states that
something is not mentioned, while 64 rejected responses do. On answerable rows the objective
therefore pushes against such statements, and the trained model nevertheless makes more of them.

Two candidate sources remain. The first is the partial and unanswerable pairs. In all 84 partial
pairs the preferred response has two sentences, one answering and one stating what is missing or
declined, and in 291 partial and unanswerable pairs the rejected response is a direct answer, in
most cases a single sentence. A model trained to disfavor the one-sentence answer on those rows
may carry that preference over to answerable rows. [Week 9](week9.md) tests this.

The second concerns why an added sentence is sometimes false. Its form follows the training
answers the row most resembles. Rows with a second, unrelated figure have training answers that
end by attributing that figure, as the added sentence of `ex_0141` does. The six rows with false
statements combine evidence that contradicts common knowledge with a detail the question does not
ask about, and in training that combination occurs almost only in partial rows whose answers end
in "is not mentioned". The model appears to borrow that ending. This account is consistent with
the data and has not been tested.

## How to run

Training, inference and judging use the commands in [Week 7](week7.md#how-to-run), with the data
files of this week and two new options:

```bash
# --seed on both training scripts
python -m grounded_refusal.train.train_sft --data data/data_v3c_train.jsonl --seed 44
python -m grounded_refusal.train.train_dpo --data data/preference_v3c_train.jsonl \
  --sft-adapter checkpoints/<sft-run> --seed 44

# --correctness-output on run_eval runs the second judge
python -m grounded_refusal.eval.run_eval --input <inference.jsonl> --output <judge.jsonl> \
  --correctness-output <correctness.jsonl>
```

`run_eval.py` prints the metrics of one model. The means over seeds and the count of responses
with an added sentence come from one-off scripts that are not in the repository.

## Next

Three questions follow from these results, in the order the results raise them. The first is
whether DPO lowers the probability of ending the response after the answer, as the account of
section 4 predicts, and whether the drop is largest where the evidence contains unused material;
both can be measured on the existing models without training. The second is the conditions under
which an added sentence is false: six rows are too few to separate the two suspected conditions,
and test rows that hold the question fixed and vary only the evidence would separate them. The
third is automatic scoring. A refusal, a partial answer and a false statement about what the
evidence lacks each assert that the evidence lacks something, and a judge item that checks such
assertions would replace the manual count.

## Appendix: per-seed values

The tables above report means. The values for each model are listed in the order of seeds 42, 44,
45, 46.

| | `v3a` SFT | `v3a` DPO | `v3c` SFT | `v3c` DPO |
|---|---|---|---|---|
| Unanswerable rows correctly refused, of 41 | 39, 38, 39, 39 | 39, 39, 40, 38 | 39, 39, 39, 40 | 40, 40, 38, 39 |
| Answerable rows wrongly refused, of 60 | 1, 1, 1, 2 | 2, 1, 3, 3 | 2, 3, 4, 3 | 2, 2, 4, 3 |
| Partial rows handled correctly, of 19 | 15, 15, 14, 17 | 17, 15, 13, 16 | 19, 19, 19, 18 | 19, 19, 19, 18 |
| Rows answered correctly, of 79 | 69, 70, 69, 67 | 70, 70, 67, 69 | 71, 71, 70, 71 | 72, 72, 70, 70 |
| `correct_but_unfaithful` | 1, 3, 3, 3 | 2, 4, 6, 5 | 3, 3, 1, 1 | 5, 5, 4, 7 |
| Responses with an added sentence, of 60 | 7, 10, 10, 13 | 27, 16, 21, 23 | 15, 11, 9, 13 | 26, 22, 21, 19 |
