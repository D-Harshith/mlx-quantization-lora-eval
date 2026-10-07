# On-Device Report Card: Llama 3.2 3B at three precisions

October 2026. Apple M5 with 16 GB RAM, MLX 0.32.3, mlx-lm 0.32.0.

**Built with Llama.** The fine-tuned adapter in `adapters/4bit_lora` is named Llama-3.2-3B-issue-triage-LoRA. It is derived from Llama 3.2 and distributed under the Llama 3.2 Community License; see `adapters/4bit_lora/LICENSE` and `NOTICE`.

## Result

Shrinking Llama 3.2 3B from 16-bit to 4-bit weights cost no measurable accuracy on GitHub issue classification. The 4-bit model used 2.5 GB of memory instead of 6.9 GB and generated text 3.5 times faster. A LoRA fine-tune of the 4-bit model then raised accuracy from 67.0% to 81.7%, almost entirely by fixing the question label. Merging that adapter into the 4-bit weights gave back most of the gain, ending at 70.0%.

## Setup

- **Task:** label a GitHub issue as bug, feature or question from its title and body.
- **Data:** the NLBSE issue-report dataset, covering react, tensorflow, vscode, bitcoin and opencv. The eval set is 300 test issues, 100 per label, drawn with a fixed seed and frozen before any run.
- **Method:** one fixed prompt, greedy decoding, issue body cut to 1,500 characters, one process per variant.

| Variant | Accuracy | Bug | Feature | Question | Seconds per issue | Peak memory | Tokens/sec |
|---|---|---|---|---|---|---|---|
| 16-bit float (unquantized) | 66.3% ± 5.3 | 83% | 86% | 30% | 0.51 | 6.9 GB | 17.5 |
| 8-bit | 66.3% ± 5.3 | 83% | 87% | 29% | 0.42 | 4.0 GB | 34.7 |
| 4-bit | 67.0% ± 5.3 | 92% | 81% | 28% | 0.31 | 2.5 GB | 61.7 |
| 4-bit + LoRA | 81.7% ± 4.4 | 80% | 84% | 81% | 0.42 | 2.9 GB | not measured |
| 4-bit + LoRA, merged | 70.0% ± 5.2 | 90% | 80% | 40% | 0.32 | 2.5 GB | not measured |

The ± figure is the 95% margin of error over 300 issues. Tokens/sec comes from a separate benchmark (400-token prompt, 200 generated tokens, 10 trials), because a one-word answer says nothing about generation speed.

## Are the differences real?

Every variant answered the same 300 issues, so they can be compared issue by issue with an exact McNemar test.

| Comparison | Only the first is right | Only the second is right | p-value |
|---|---|---|---|
| 16-bit vs 8-bit | 1 | 1 | 1.00 |
| 4-bit vs 8-bit | 12 | 10 | 0.83 |
| 4-bit vs 16-bit | 12 | 10 | 0.83 |
| 4-bit vs 4-bit + LoRA | 21 | 65 | < 0.001 |
| 4-bit + LoRA vs merged | 54 | 19 | < 0.001 |
| 4-bit vs merged | 3 | 12 | 0.035 |

- The three precisions are tied on accuracy, and the LoRA gain is real.
- Tied does not mean identical. The 4-bit model answers "bug" for 160 of the 300 issues, against 140 for the other two. Compared with 8-bit it gets 9 more bugs right and 6 fewer features (p = 0.004 on bugs alone, not corrected for the number of comparisons).

## What the fine-tune changed

The adapter is rank 8 on the last 16 layers: 6.9 million trainable weights, 0.2% of the model, saved as a 28 MB file. It was trained for one pass over 1,350 training-file issues (450 per label), scoring only the label word, in about 23 minutes.

Answers on the 100 true questions:

| Model's answer | 4-bit | 4-bit + LoRA |
|---|---|---|
| question | 28 | 81 |
| bug | 53 | 7 |
| feature | 19 | 12 |

- The base model got 72 of the 100 questions wrong, calling 53 of them bugs. The fine-tune fixed 53 of those 72 and lost none of the 28 it already had right.
- The cost is on bugs: accuracy fell from 92% to 80%, because 17 bugs are now called questions, up from 2 (p = 0.012).
- It is also slower, 0.42 against 0.31 seconds per issue, with the adapter loaded as separate layers and not merged into the weights.

## Merging the adapter

Merging (fusing) the adapter into the model's weights removes the extra layers, so the model runs at base speed again: 0.32 seconds per issue and 2.5 GB. It also lost most of the fine-tune.

- Accuracy fell from 81.7% to 70.0%, and questions from 81% to 40%. The merged model is clearly worse than the separate adapter and only slightly better than the base.
- Its answers match the base model's on 285 of the 300 issues.
- The likely cause is how mlx-lm merges into a quantized model: it adds the adapter to the weights, then rounds them back to 4 bits. On this setup the choice is accuracy with the adapter kept separate, or speed with it merged.

## Checks

- **Truncation:** on 150 long issues from the training file, a 1,500-character cap scored 88 correct. A 6,000-character cap scored 84 and took twice as long. The cap was chosen on training data, never on the eval set.
- **Duplicates:** no eval issue has the same title and body as a training issue. Eight share only an unfilled issue-template body with one. The base model got 7 of those 8 right and the fine-tune 8, and dropping them leaves 66.4% against 81.2% on 292 issues.
- **Title prefixes:** 8 of the 300 titles start with a prefix such as "Bug:". The base models followed the prefix in all 8, and the dataset label disagrees with it in 4, so the most this could move accuracy is 2.7 points.

## Limits

- One model, one task and one machine. With 300 issues, differences under about 5 points cannot be told apart.
- The dataset's labels are noisy, as the 4 prefix disagreements show, which caps the accuracy any model can reach.
- Tokens/sec is not measured for the two LoRA variants, and merging the adapter into a full-precision copy of the model was not tried.

## Reproduce

Run `scripts/LLM_eval.py` once per variant and `scripts/benchmark.py` for tokens/sec. The combined table is `results/comparison.csv`.
