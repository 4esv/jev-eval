# jev-eval

Independent measurement of [TypeSafe Jev](https://typesafe.ai) against GPT-5.6 Terra on three public labelled classification tasks: accuracy, calibration, latency, cost.

## Results

300 items per task, run 2026-09-17 with `jev-1.13.0` and `openai/gpt-5.6-terra` via OpenRouter.

| | intent (77 classes) | sentiment (5 levels) | positive/negative |
|---|---|---|---|
| Accuracy, Jev / Terra | 0.78 / 0.85 | 0.57 / 0.59 | 0.97 / 0.97 |
| Gap beyond noise (95% CI) | borderline | no | no |
| Calibration error (ECE), Jev / Terra | 0.11 / 0.08 | 0.20 / 0.30 | 0.04 / 0.02 |
| Median latency, Jev / Terra | 0.20 s / 1.04 s | 0.19 s / 1.06 s | 0.20 s / 1.04 s |
| Cost per 1k calls, Jev / Terra | $0.04 / $2.02 | $0.01 / $0.64 | $0.02 / $0.85 |

- Speed 5x, cost 41–50x per call. Advertised: 193x and 444x.
- Jev counts about twice the input tokens for the same text (340 vs 162 on identical sentences).
- Accuracy equal on the two easy tasks; 6.7 points lower on 77-way routing.
- Calibration better than Terra's on one task, worse on two. AUROC of confidence against correctness: 0.83 / 0.61 / 0.94 (Jev), 0.81 / 0.64 / 0.96 (Terra).
- Not deterministic: identical inputs changed the label on 1.7% (intent) and 3.3% (sentiment) of items.
- "Zero hallucinations" means every answer is a listed option. True for all 1,800 Jev calls; also true for all 1,800 Terra calls under a strict JSON schema.
- Terra with `reasoning.effort = "medium"` used 4–17 reasoning tokens on these inputs, answered in about 1 s, and was not more accurate.

Full tables, threshold coverage and the reasoning run: [`results/summary.md`](results/summary.md).

## Run

```bash
cp .env.example .env        # TYPESAFE_API_KEY, OPENROUTER_API_KEY
uv sync
uv run python -m evaljev.run --model jev --n 300
uv run python -m evaljev.run --model terra --n 300 --cap 10
uv run python -m evaljev.run --model terra-reason --n 300 --cap 10
uv run python -m evaljev.run --model jev --n 300 --tag rerun
uv run python -m evaljev.report
uv run pytest
```

Runs resume; delete `results/<task>/<model>.jsonl` to redo a pair. `--cap` stops at that many dollars of OpenRouter spend. The full run cost $0.045 on Jev and $2.10 on OpenRouter.

## Notes

| task | data | Jev question | Terra output |
|---|---|---|---|
| `intent` | Banking77 test (`mteb/banking77`), 3–4 per class | `choice`, 77 options | JSON `{label, confidence}` |
| `sentiment5` | SST-5 test (`SetFit/sst5`), 60 per level | `score`, 5 ordered levels | JSON `{label, confidence}` |
| `polarity` | IMDB test, reviews ≤ 300 words, 150 per class | `noul` | JSON `{label, confidence}` |

- Samples are seeded and stratified (`evaljev/data.py`) and committed in `data/`.
- Both models get the same instruction and option names. Terra's `label` is a schema enum.
- Confidence is the probability of the chosen label: Jev's returned distribution; Terra's stated number.
- Latency is the wall-clock time of the successful HTTP request, same clock for both.
- Cost: Jev `input_tokens × $0.042/M`, output free; Terra as billed by OpenRouter.
- Raw records with request ids: `results/<task>/<model>.jsonl`. Metrics: `evaljev/metrics.py`, tested in `tests/`.
- Caveats: the datasets are public and old; Terra's confidence is self-reported and clusters at 0.98–0.99; one run from one machine on one day; no prompt tuning; at n = 300, differences under about 5 points are noise; the Banking77 mirror has 3,076 test rows against 3,080 in the original.
