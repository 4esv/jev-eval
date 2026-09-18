# jev-eval

[Jev](https://typesafe.ai) is a paid classification API from TypeSafe. You send it some text and a list of allowed answers. It returns one answer and a probability for each option. It cannot return free text. It answers in about 200 ms and charges $0.042 per million input tokens.

TypeSafe claims Jev is as accurate as frontier LLMs on this kind of decision, 193x faster, 444x cheaper, and honest enough about its confidence that software can act on it without a human.

This repo tests those claims against OpenAI's GPT-5.6 Terra on three public datasets with known correct answers. TypeSafe's own evaluation measures how often Jev agrees with other LLMs on tests its team wrote. This one measures accuracy against labels.

The run was on 2026-09-17 with `jev-1.13.0` and `openai/gpt-5.6-terra` through OpenRouter. Each task has 300 items. That is 2,700 scored calls, plus 900 repeat calls to Jev to check whether it gives the same answer twice.

## Results

| | intent (77 options) | sentiment (5 levels) | positive/negative |
|---|---|---|---|
| **Accuracy** Jev / Terra | 0.78 / **0.85** | 0.57 / 0.59 | 0.97 / 0.97 |
| Is the gap larger than noise? | borderline | no | no |
| **Calibration error** (lower is better) Jev / Terra | 0.11 / **0.08** | **0.20** / 0.30 | 0.04 / **0.02** |
| **Median latency** Jev / Terra | 0.20 s / 1.04 s | 0.19 s / 1.06 s | 0.20 s / 1.04 s |
| **Cost per 1k calls** Jev / Terra | $0.04 / $2.02 | $0.01 / $0.64 | $0.02 / $0.85 |

Calibration error measures whether a model that says 0.9 is right 90% of the time. A score of 0 means the confidence numbers are exactly honest.

The full tables and a run of Terra in reasoning mode are in [`results/summary.md`](results/summary.md).

## Findings

Jev was about 5x faster than Terra and 41x to 50x cheaper per call. The advertised numbers are 193x and 444x. One reason the cost gap is smaller than the price list suggests is that Jev counts about twice as many input tokens for the same text. On identical sentences it counted 340 tokens and Terra counted 162.

Accuracy was tied on the two easy tasks. On the 77-way routing task Jev was 6.7 points behind. With 300 items that gap is at the edge of what the test can resolve.

Jev's probabilities were not reliably calibrated. They were better than Terra's on the 5-level sentiment task and worse on the other two. On the routing task, answers given with 0.98 confidence were right 90% of the time.

Jev's confidence was no better than Terra's at flagging its own mistakes. Both models separate right answers from wrong ones about equally well, so neither gives a better signal for deciding when to escalate to a human. The scores are in the summary file.

Jev is not deterministic. The same input produced a different answer 1.7% of the time on routing and 3.3% of the time on sentiment. It was 0% on positive/negative.

"Zero hallucinations" means the answer is always one of the allowed options. That was true for all 1,800 Jev calls. It was also true for all 1,800 Terra calls, because a strict JSON schema enforces the same thing.

Reasoning mode did not change the picture. TypeSafe's 8.6 second Terra figure used reasoning. On these short inputs Terra with reasoning used 4 to 17 reasoning tokens, answered in about 1 second, and was no more accurate.

## If you are deciding whether to use it

Jev fits high-volume yes/no or few-way classification where an LLM call is too slow or too expensive. On the positive/negative task it scored 0.97 accuracy at 0.2 seconds and $0.02 per 1k calls.

Pick your confidence threshold from your own labelled data. The probabilities were not reliably calibrated here, and an LLM's self-reported confidence worked about as well.

Do not assume the same input gives the same answer. It differed a few percent of the time.

Routing across dozens of options lost several points to the LLM. Test that case on your own data. Adding a description to each option was not tried and might close the gap.

![reliability](results/reliability.png)

![coverage](results/coverage.png)

## Method

| task | data | Jev question type | Terra output |
|---|---|---|---|
| `intent` | Banking77 test (`mteb/banking77`), 300 items, 3 or 4 per class | `choice` with 77 options | JSON `{label, confidence}` |
| `sentiment5` | SST-5 test (`SetFit/sst5`), 60 per level | `score` with 5 ordered levels | JSON `{label, confidence}` |
| `polarity` | IMDB test, reviews of 300 words or fewer, 150 per class | `noul` (yes/no) | JSON `{label, confidence}` |

The samples are seeded and balanced across classes by `evaljev/data.py` and committed in `data/`.

Both models get the same instruction sentence and the same option names. Terra's `label` field is a schema enum of those options.

Confidence means the probability of the chosen answer. For Jev it comes from the distribution it returns. For Terra it is the number the model writes in the JSON.

Latency is the wall-clock time of the successful HTTP request, measured the same way for both models.

Jev's cost is input tokens times $0.042 per million, because output is free. Terra's cost is what OpenRouter billed.

Every request and raw response is in `results/<task>/<model>.jsonl`.

## Caveats

All three datasets are public and old. Either model may have trained on them.

Terra's confidence is self-reported and clusters at 0.98 to 0.99. Jev's is a real distribution. That should favour Jev on calibration.

This is one run from one machine on one day. TypeSafe serves from the US West Coast and OpenRouter adds a network hop.

Neither model's prompt was tuned. The 77 routing options were bare names with no descriptions.

With 300 items per task, differences under about 5 points are noise.

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

Runs resume. Items already scored are skipped and failed items are retried. Delete `results/<task>/<model>.jsonl` to redo that pair. The full run cost $0.045 on Jev and $2.10 on OpenRouter.
