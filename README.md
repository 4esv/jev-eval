# jev-eval

[Jev](https://typesafe.ai) is a paid API. You send it text plus a fixed list of allowed
answers; it returns one answer and a probability for each option. No free text. It answers in
about 200 ms and charges $0.042 per million input tokens. TypeSafe says it matches frontier
LLMs on this kind of decision while being "193x faster, 444x cheaper", and that its
probabilities are honest enough to auto-act on.

This repo checks that against OpenAI's GPT-5.6 Terra on three public datasets with known
correct answers. TypeSafe's own numbers measure agreement with other LLMs on tests its team
wrote; these measure accuracy against labels.

Run on 2026-09-17: `jev-1.13.0` and `openai/gpt-5.6-terra` (via OpenRouter). 300 items per
task, 2,700 scored calls, plus 900 repeat calls to Jev.

## Results

| | intent (77 options) | sentiment (5 levels) | positive/negative |
|---|---|---|---|
| **Accuracy** Jev / Terra | 0.78 / **0.85** | 0.57 / 0.59 | 0.97 / 0.97 |
| Gap larger than noise? | borderline | no | no |
| **Calibration error** (lower is better) Jev / Terra | 0.11 / **0.08** | **0.20** / 0.30 | 0.04 / **0.02** |
| **Median latency** Jev / Terra | 0.20 s / 1.04 s | 0.19 s / 1.06 s | 0.20 s / 1.04 s |
| **Cost per 1k calls** Jev / Terra | $0.04 / $2.02 | $0.01 / $0.64 | $0.02 / $0.85 |

Calibration error: if a model says 0.9, is it right 90% of the time? 0 means yes exactly.
Full tables, per-threshold coverage and a reasoning-mode Terra run: [`results/summary.md`](results/summary.md).

## Findings

- **5x faster and 41–50x cheaper per call.** Real, but not 193x and 444x. Jev also counts
  about twice the input tokens for the same text (340 vs 162 on identical sentences), which
  eats part of the per-token price gap.
- **Accuracy: tied on the easy tasks, 6.7 points behind on 77-way routing.** That gap is at
  the edge of what 300 items can resolve.
- **Probabilities are not reliably calibrated.** Better than Terra on 5-level sentiment,
  worse on the other two. On intent, answers Jev gives at 0.98 confidence are right 90% of
  the time.
- **Its confidence is no better than an LLM's at flagging its own mistakes.** Used as an
  escalate-to-human signal, Jev's probability separates right from wrong answers about as
  well as Terra's self-reported number (AUROC 0.83 / 0.61 / 0.94 vs 0.81 / 0.64 / 0.96).
- **Not deterministic.** The same input got a different answer 1.7% of the time on intent and
  3.3% on sentiment. 0% on positive/negative.
- **"Zero hallucinations" means "always returns a valid option".** True for all 1,800 Jev calls.
  Also true for all 1,800 Terra calls, because a strict JSON schema does the same job.
- **Reasoning mode didn't change the picture.** TypeSafe's 8.6 s Terra figure used reasoning.
  On these short inputs, Terra with reasoning spent 4–17 reasoning tokens, answered in about
  1 s, and was no more accurate.

## If you're deciding whether to use it

- Good fit: high-volume yes/no or few-way classification where a 1 s LLM call is too slow or
  too expensive. Here: 0.97 accuracy, 0.2 s, $0.02 per 1k calls.
- Threshold on confidence, but measure the threshold on your own labelled data first. The
  probabilities were not reliably calibrated here, and an LLM's stated confidence worked
  about as well.
- Don't assume repeatability. Same input, different answer, a few percent of the time.
- Routing across dozens of options lost several points to the LLM. Test that case on your
  data. Per-option descriptions were not tried and might close the gap.

![reliability](results/reliability.png)

![coverage](results/coverage.png)

## Method

| task | data | Jev question type | Terra output |
|---|---|---|---|
| `intent` | Banking77 test (`mteb/banking77`), 300 items, 3–4 per class | `choice`, 77 options | JSON `{label, confidence}` |
| `sentiment5` | SST-5 test (`SetFit/sst5`), 60 per level | `score`, 5 ordered levels | JSON `{label, confidence}` |
| `polarity` | IMDB test, reviews ≤300 words, 150 per class | `noul` (yes/no) | JSON `{label, confidence}` |

- Samples are seeded and balanced across classes (`evaljev/data.py`), committed in `data/`.
- Both models get the same instruction sentence and option names. Terra's `label` is a schema enum.
- Confidence = probability of the chosen answer. Jev's comes from its returned distribution;
  Terra's is the number it writes in the JSON.
- Latency = wall-clock time of the successful HTTP request, same clock for both.
- Cost: Jev `input_tokens × $0.042/M` (output is free); Terra as billed by OpenRouter.
- Every request and raw response is in `results/<task>/<model>.jsonl`.

## Caveats

- All three datasets are public and old. Either model may have trained on them.
- Terra's confidence is self-reported and clusters at 0.98–0.99; Jev's is a real distribution.
  That should favour Jev on calibration.
- One run, one machine, one day. TypeSafe serves from the US West Coast; OpenRouter adds a hop.
- No prompt tuning on either side. The 77 intent options were bare names with no descriptions.
- n = 300 per task: differences under about 5 points are noise.

## Rerun it

```bash
cp .env.example .env        # fill in TYPESAFE_API_KEY and OPENROUTER_API_KEY
uv sync
uv run python -m evaljev.run --model jev --n 300
uv run python -m evaljev.run --model terra --n 300 --cap 10  # stops at $10 of OpenRouter spend
uv run python -m evaljev.run --model terra-reason --n 300 --cap 10
uv run python -m evaljev.run --model jev --n 300 --tag rerun # determinism check
uv run python -m evaljev.report                               # results/summary.md + plots
uv run pytest
```

Runs resume: scored ids are skipped, failed ones retried. Delete `results/<task>/<model>.jsonl`
to redo a pair. The full run cost $0.045 on Jev and $2.10 on OpenRouter.
