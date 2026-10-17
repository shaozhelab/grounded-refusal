# The base model compared with SFT and DPO

The base model, Qwen2.5-3B-Instruct without fine-tuning, answers 71 of the 79 answerable and
partial held-out rows correctly, as many as the SFT and DPO models. What fine-tuning adds is the
handling of questions the evidence cannot fully answer: correct refusals rise from 31 to 39.3 of
41 unanswerable rows, correct partial answers from 9 to 18.8 of 19, and the hallucination rate
falls from 0.218 to 0.087.

The base model also writes more than one sentence on 23 of the 60 answerable rows, close to the 22
of the DPO models, which raises the question of whether DPO returns the model to its behavior
before fine-tuning. It does not. The rows on which DPO adds a sentence are not those on which the
base model does, no added sentence is shared, and the phrasings most typical of DPO never occur in
the base model's responses. The sentence DPO adds is a behavior produced by the DPO stage.

| File | Contents |
|------|----------|
| `outputs/inference-qwen2.5-3b-instruct/base_v3b_heldout.jsonl` | The base model's 120 held-out responses with the `v3b` labels (#28) |
| `outputs/eval-qwen2.5-3b-instruct/base_v3b_heldout_eval_judge-gpt5-mini.jsonl` | `predicted_behavior` and `is_faithful` for each response |
| `outputs/eval-qwen2.5-3b-instruct/base_v3b_heldout_correctness_judge-gpt5-mini.jsonl` | `answer_match` for the 79 answerable and partial rows |

## Base, SFT and DPO

The SFT and DPO models are the eight `data_v3c_train` models of #23, at seeds 42, 44, 45 and 46,
each DPO model trained from the SFT model of the same seed. Their responses and judge outputs are
in `outputs/inference-qwen2.5-3b-instruct-{sft,dpo}/` and `outputs/eval-qwen2.5-3b-instruct-{sft,dpo}/`.
All three are scored on the same 120 rows against the `data_v3b_heldout.jsonl` labels: 60
answerable, 19 partial and 41 unanswerable. The SFT and DPO columns are means over the four seeds;
the base model is a single model.

| Metric | Meaning | Base | SFT | DPO |
|---|---|---:|---:|---:|
| `abstention_recall` | Share of unanswerable rows refused, higher is better | 0.756 | 0.957 | 0.957 |
| `abstention_precision` | Share of refusals that were correct, higher is better | 0.969 | 0.929 | 0.935 |
| `over_refusal_rate` | Share of answerable rows refused, lower is better | 0.017 | 0.050 | 0.046 |
| `hallucination_rate` | Share of attempted answers that are unfaithful, lower is better | 0.218 | 0.087 | 0.125 |
| `partial_match_rate` | Share of partial rows answered in part, higher is better | 0.474 | 0.987 | 0.987 |
| `answer_accuracy` | Share of answerable and partial rows answered correctly, higher is better | 0.899 | 0.896 | 0.899 |

The same results as row counts:

| | Base | SFT | DPO |
|---|---:|---:|---:|
| Unanswerable rows correctly refused, of 41 | 31 | 39.3 | 39.3 |
| Answerable rows wrongly refused, of 60 | 1 | 3.0 | 2.8 |
| Partial rows handled correctly, of 19 | 9 | 18.8 | 18.8 |
| Rows answered correctly, of 79 | 71 | 70.8 | 71.0 |
| `correct_and_faithful` | 61 | 68.8 | 65.8 |
| `correct_but_unfaithful` | 9 | 2.0 | 5.3 |
| `wrong_but_faithful` | 3 | 2.0 | 1.8 |
| `wrong_and_unfaithful` | 4 | 3.0 | 3.3 |
| `refused` | 2 | 3.3 | 3.0 |

Answer accuracy is 0.899 for the base model and 0.896 for SFT, so fine-tuning does not make the
model answer more accurately. It teaches the model to refuse what the evidence does not support,
raising correct refusals from 31 to 39.3 of 41 at the cost of about two more wrongly refused
answerable rows, and to answer only the supported part of a partial question, raising correct
partial answers from 9 to 18.8 of 19.

Fine-tuning also lowers the hallucination rate from 0.218 to 0.087, on every kind of row. The base
model gives 19 unfaithful responses: 9 on answerable rows, 6 on unanswerable rows it answered
instead of refusing, and 4 on partial rows. An SFT model gives 5.0, 1.75 and 0. DPO leaves
accuracy and refusals unchanged but raises the hallucination rate to 0.125, and the increase falls
entirely on answerable rows, from 5.0 to 8.25 unfaithful responses. In terms of answer outcomes,
these are correct answers accompanied by an unsupported statement: `correct_but_unfaithful` rises
from 2.0 to 5.3.

## Does the added sentence come from the base model?

A DPO model often continues after answering. On `ex_0094`, which asks only for the temperature at
which water freezes, all three models give the same answer, and only the DPO model continues:

| | Response to `ex_0094`, seed 46 |
|---|---|
| Base | Water freezes at 10 degrees Celsius at standard atmospheric pressure, as stated in the evidence. |
| SFT | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. |
| DPO | According to the evidence, water freezes at 10 degrees Celsius at standard atmospheric pressure. **The rest of the question -- about the monitoring system triggering an alert -- isn't mentioned.** |

Counted as in [Week 8](reports/week8.md), as responses to answerable rows with two or more
sentences, the base model writes such a response on 23 of the 60 rows, SFT models on 12.0 and DPO
models on 22.0. The similar counts for the base model and DPO suggest that DPO may return the model
to its behavior before fine-tuning. Three comparisons show that it does not.

**The rows differ.** Over the four seeds there are 44 responses in which the DPO model adds a
sentence and the SFT model of the same seed does not. The base model also adds a sentence on 13 of
these rows, 30%, which is lower than its rate of 38% over all 60 answerable rows.

**The sentences differ.** On the rows where both the base model and a DPO model add a sentence, no
added sentence of the DPO model is identical to one of the base model.

**The phrasings differ.** The two phrasings most typical of DPO never occur in the base model's
responses. The table counts responses to answerable rows whose added sentences contain each phrasing,
over the four SFT and the four DPO models.

| Phrasing | Base | SFT, 4 models | DPO, 4 models |
|---|---:|---:|---:|
| "The rest of the question / statement / sentence -- … -- isn't mentioned" | 0 | 2 | 14 |
| "The X figure belongs to the (unrelated) Y" | 0 | 2 | 11 |

### What the base model adds

The 23 added sentences of the base model fall into six kinds.

| Kind | Count | Rows | Example of the added text |
|---|---:|---|---|
| Reasoning or calculation: the first sentence states a relation and the answer is computed after it | 6 | `ex_0109` `ex_0110` `ex_0118` `ex_0120` `ex_0124` `ex_0125` | "The almanac states that Jupiter has 3 moons. Therefore, Saturn has 3 × 5 = 15 moons." |
| A comment on what the evidence mentions | 7 | `ex_0092` `ex_0098` `ex_0122` `ex_0133` `ex_0135` `ex_0136` `ex_0138` | "The evidence does not mention anything about it sitting on the Pacific coastline." |
| A correction of the evidence from outside knowledge | 3 | `ex_0193` `ex_0601` `ex_0609` | "However, this contradicts historical facts as the Rosetta Stone is actually inscribed in three ancient scripts …" |
| A pointer to the source | 3 | `ex_0123` `ex_0142` `ex_0650` | "The evidence directly states this figure." |
| A restatement of the answer | 2 | `ex_0095` `ex_0633` | "Question: Who served as the first President of the United States? Answer: Thomas Jefferson." |
| Hesitation or self-contradiction after the answer | 2 | `ex_0143` `ex_0351` | Having answered that Mozart lived to 90: "I do not have information about how old Mozart was when he died." |

The base model's additions are varied. In six of them the first sentence does not yet contain the
answer, so the second sentence completes the answer rather than adding to it. Ten comment on the
evidence, seven on what it mentions and three on whether it is true, but in the base model's own
words: "The evidence does not mention …", "However, …".

### What DPO adds

The 88 added sentences of the four DPO models are labeled by kind in
[`TRAILING_LABELS.md`](TRAILING_LABELS.md).

| Kind | Count | Share |
|---|---:|---:|
| A comment on the evidence or the question (`evidence_comment`) | 61 | 69% |
| A continuation of the answer or the calculation (`answer`) | 14 | 16% |
| Other, such as a pointer to the source or a refusal (`other`) | 13 | 15% |

DPO's additions are concentrated. Most comment on the evidence after the answer is complete, often
in the two phrasings of the table above. Four repeat an instruction embedded in the evidence, that
"the 2022 figure is outdated".

### Where the phrasings come from

| Stage | What changes | Evidence |
|---|---|---|
| Base to SFT | SFT removes the base model's habit of continuing to reason and of correcting the evidence from outside knowledge, and learns the wording of its training targets | 272 of the 323 answerable SFT targets are a single sentence. 92 contain the clause "the X figure belongs to …", and 34 of the 84 partial targets end with "… isn't stated" |
| SFT to DPO | DPO lowers the probability of ending the response after the first sentence. When the model does not stop, it continues in the wording learned in SFT | The stop probability falls from 0.81 to 0.65 ([Week 9](reports/week9.md), Ablation 4); "The rest of the question" rises from 2 responses to 14 |

The phrase "The rest of the question" appears nowhere in the training data, neither in the SFT
targets nor in the chosen or rejected responses of the preference pairs. The SFT models compose it
from the "… isn't stated" endings of the partial targets and use it rarely; DPO makes it common.
Ablation 2 of Week 9 shows that it does not depend on the partial pairs: DPO models trained only on
answerable pairs still write it, on 2 to 5 rows each.

The base model writes more because it is still reasoning toward the answer, or comments on the
evidence in its own words and from its own knowledge. A DPO model writes more after the answer is
complete, commenting on the evidence in a phrasing learned in SFT and amplified by DPO. The added
sentence is therefore produced by the DPO stage, not inherited from the base model. On the base
model, two or more sentences often mean an answer still being computed, so its count of 23 does not
measure the same behavior as the SFT and DPO counts.

## How the base results were obtained

The base model's responses are those of [Week 7](reports/week7.md), scored there against the
`v2` labels. The `v2` and `v3b` held-out files contain the same 120 rows with identical prompts;
`v3b` changes the label and reference answer of 7 rows, all previously unanswerable, 5 of them to
partial and 2 to answerable. The responses are therefore reused and joined by row id to the `v3b`
labels. The behavior and faithfulness judge reads only the prompt and the response, so its Week 7
output is reused as well. The correctness judge, which reads the reference answer, is run on the
79 answerable and partial rows with gpt-5-mini.

| Check | Result |
|---|---|
| The reused responses match the current inference code | 12 regenerated responses are identical to the stored ones |
| The behavior and faithfulness judge is the same | `judge.py` has changed since the Week 7 run only in a comment |
| The correctness judge is the same | `correctness.py` has not changed since 10 October, before any of the three models was scored |
| The prompts are the same | The 120 prompts are identical character for character |

## How to reproduce

The base column is recomputed from the committed judge outputs, without calling the API. The SFT
and DPO columns are recomputed in the same way from their files, as in [Week 8](reports/week8.md).

```bash
python -m grounded_refusal.eval.run_eval --resume \
  --input outputs/inference-qwen2.5-3b-instruct/base_v3b_heldout.jsonl \
  --output outputs/eval-qwen2.5-3b-instruct/base_v3b_heldout_eval_judge-gpt5-mini.jsonl \
  --correctness-output outputs/eval-qwen2.5-3b-instruct/base_v3b_heldout_correctness_judge-gpt5-mini.jsonl
```
