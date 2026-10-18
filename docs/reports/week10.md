# Week 10 report: REM-DPO and the base-model baseline

## Summary

Week 9 traced the sentences DPO adds after a complete answer to the DPO loss, which lowers the
probability of stopping after the first sentence, and left open how the loss does so. This week
answers that question and removes the effect. LD-DPO, a DPO variant designed to remove length bias,
leaves the added sentences unchanged (89 answerable responses with two or more sentences, against
88 for DPO), so they do not come from the length of the responses
([#29](https://github.com/stxxllbu/grounded-refusal/pull/29)). Splitting DPO's change to each
response by token shows that it falls mostly on the EOS token of the rejected responses
([#30](https://github.com/stxxllbu/grounded-refusal/pull/30)). Because every response ends with the
same EOS token, pushing it down in the rejected responses lowers it in all responses, and the model
stops less often. REM-DPO, DPO with the rejected EOS token masked out of the loss, restores the
probability of stopping after the first sentence from 0.65 to SFT's 0.81 and the responses with
added sentences from 88 to SFT's 48. Its hallucination rate falls from 0.125 to 0.106, while
refusals, partial answers and accuracy stay within about one row of SFT and DPO
([#31](https://github.com/stxxllbu/grounded-refusal/pull/31)).

The week also adds the base model as a baseline. SFT and DPO clearly outperform it in refusals,
partial answers and hallucination, without any loss of answer accuracy, and the sentence DPO adds is
not inherited from it ([`BASE_MODEL.md`](../BASE_MODEL.md)).

## Development

| Work | What was added | Files | PR |
|---|---|---|---|
| Baseline | The base model scored against the `v3b` held-out labels | [`base_v3b_heldout.jsonl`](../../outputs/inference-qwen2.5-3b-instruct/base_v3b_heldout.jsonl) and its two judge outputs, [`BASE_MODEL.md`](../BASE_MODEL.md) | [#28](https://github.com/stxxllbu/grounded-refusal/pull/28) |
| Ablation | LD-DPO at `ld_alpha` 0.5, 4 seeds | `configs/train/dpo_ld.yaml`, branch `ablation/dpo-length-desensitized` | [#29](https://github.com/stxxllbu/grounded-refusal/pull/29) |
| Ablation | Per-token change in log-probability from SFT to DPO | `token_attribution.py`, `token_attribution_summary.py`, branch `ablation/token-attribution` | [#30](https://github.com/stxxllbu/grounded-refusal/pull/30) |
| REM-DPO | DPO with the rejected EOS token masked out of the loss, 4 seeds, both judges | `MaskRejectedEndCollator` in `train_dpo.py`, `configs/train/dpo_mask_rejected_end.yaml`, branch `ablation/mask-rejected-end` | [#31](https://github.com/stxxllbu/grounded-refusal/pull/31) |

The ablations and REM-DPO are draft pull requests kept on their own branches. Adapter weights are not
committed to git, consistent with [`checkpoints/README.md`](../../checkpoints/README.md).

## 1. The base model as a baseline

The base model's responses from Week 7 are scored against the `v3b` labels used for the SFT and DPO
models of Week 8, so the three appear on one table. Both fine-tuned models clearly outperform the
base model: correct refusals of unanswerable rows rise from 31 to 39.3 of 41, correct partial
answers from 9 to 18.8 of 19, and the hallucination rate falls from 0.218 to 0.087 for SFT and 0.125
for DPO, while answer accuracy stays at about 71 of the 79 answerable and partial rows.

The base model also often writes more than one sentence, which allows the sentence DPO adds to be
compared with it. The two differ in where they appear, what they contain and how they are worded:
30 of DPO's 40 additional longer responses are on rows where the base model answers in one sentence,
the base model's habits that SFT removes do not return under DPO, and DPO's two typical phrasings
never occur in the base model's responses. Details are in [`BASE_MODEL.md`](../BASE_MODEL.md).

## 2. Ablation: length-desensitized DPO

The usual explanation for DPO producing longer responses is its sensitivity to length: the loss sums
token log-probabilities, so a longer response weighs more. LD-DPO removes this sensitivity by
down-weighting the tokens of each response beyond the length of the shorter one, here by `ld_alpha`
0.5.

LD-DPO takes effect in training, where its margin between chosen and rejected responses is 6 to 9%
smaller, but its responses do not change: it writes 89 answerable responses with two or more
sentences, against 88 for DPO and 48 for SFT, matches DPO at every seed, and 187 of its 240
answerable responses are word for word those of DPO. The added sentences therefore do not come from
DPO's sensitivity to length. Details are in
[#29](https://github.com/stxxllbu/grounded-refusal/pull/29).

## 3. Ablation: the per-token change from SFT to DPO

The DPO loss and its training logs see each response only through the sum of its token
log-probabilities. This ablation splits that sum by token: for each token of a training pair or a
held-out reference answer, it takes the log-probability under DPO minus that under SFT, the
reference model of DPO training.

The change falls on the EOS token. On the rejected responses, the EOS token falls by 5.43 in
log-probability on average, against 0.37 for other tokens, and carries 26 to 38% of the whole
decrease while making up 3% of the tokens. On the chosen responses of answerable pairs, the content
is unchanged (−0.01) while the EOS token falls (−0.06 to −0.31), and on the held-out reference
answers the EOS token falls by 1.96 to 3.02 while the first token of a second sentence becomes more
likely. Since every response ends with the same EOS token, the push on the rejected responses lowers
the probability of stopping in all of them. Details are in
[#30](https://github.com/stxxllbu/grounded-refusal/pull/30).

## 4. REM-DPO: masking the rejected EOS token

REM-DPO is DPO with one change: the EOS token of each rejected response is left out of the loss, so
the rejected response can be pushed down only through its other tokens. Everything else is as in
the full-data DPO runs. The four models are scored by both judges.

Means over four seeds; Base is a single model. Metric definitions are in
[`EVAL_METRICS.md`](../EVAL_METRICS.md).

| Metric | Base | SFT | DPO | REM-DPO |
|---|---:|---:|---:|---:|
| `abstention_recall` | 0.756 | 0.957 | 0.957 | 0.951 |
| `abstention_precision` | 0.969 | 0.929 | 0.935 | 0.945 |
| `over_refusal_rate` | 0.017 | 0.050 | 0.046 | 0.037 |
| `hallucination_rate` | 0.218 | 0.087 | 0.125 | 0.106 |
| `partial_match_rate` | 0.474 | 0.987 | 0.987 | 0.961 |
| `answer_accuracy` | 0.899 | 0.896 | 0.899 | 0.883 |

The second table compares the added sentence across the models, including LD-DPO of Section 2.

| | SFT | DPO | LD-DPO | REM-DPO |
|---|---:|---:|---:|---:|
| Answerable responses with two or more sentences, of 240 (4 models) | 48 | 88 | 89 | 48 |
| Probability of stopping after the first sentence | 0.81 | 0.65 | – | 0.81 |

REM-DPO removes the added sentence. The probability of stopping after the first sentence returns
from 0.65 to SFT's 0.81 and the answerable responses with added sentences from 88 to SFT's 48, at
every seed. On `ex_0094`, REM-DPO answers word for word as SFT, without the sentence DPO adds. The
hallucination rate falls from 0.125 to 0.106, because the correct answers followed by an unsupported
statement (`correct_but_unfaithful`) return from 5.3 to SFT's 2.0 per model. Refusals, partial
answers and accuracy stay within about one row of SFT and DPO, and training still ranks the chosen
response above the rejected one in nearly every pair. Details are in
[#31](https://github.com/stxxllbu/grounded-refusal/pull/31).

## How to run

Each pull request gives its own commands: [#29](https://github.com/stxxllbu/grounded-refusal/pull/29)
for LD-DPO, [#30](https://github.com/stxxllbu/grounded-refusal/pull/30) for the per-token change and
[#31](https://github.com/stxxllbu/grounded-refusal/pull/31) for REM-DPO.
[`BASE_MODEL.md`](../BASE_MODEL.md) gives the command for the baseline.

## Next: Week 11

The first step is to write the final report around the result of this week: SFT teaches the model
to refuse and to answer in part, DPO adds a sentence after complete answers through the shared EOS
token, and REM-DPO removes it. The second step is to place REM-DPO among prior work on DPO's length
bias and on the EOS token, and to confirm that its name is not already in use.
