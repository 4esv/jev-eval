# jev-eval

Benchmark [TypeSafe Jev](https://typesafe.ai) against any OpenRouter model or a local checkpoint on labelled classification data: accuracy, calibration, confidence distribution, latency, cost. Ships four tasks; the results below are those tasks against GPT-5.6 Terra and [Laya](https://huggingface.co/convaiinnovations/laya).

## Results

300 items per task, run 2026-09-17 and 2026-09-21. `jev-1.13.0`, `openai/gpt-5.6-terra` via OpenRouter, `convaiinnovations/laya` (421M, Apache 2.0) local on an M-series GPU.

| | intent (77 classes) | sentiment (5 levels) | polarity as `noul` | polarity as `choice` |
|---|---|---|---|---|
| **Accuracy** Jev / Laya / Terra | 0.780 / 0.370 / **0.847** | 0.570 / 0.310 / **0.593** | **0.970** / 0.507 / **0.970** | **0.967** / 0.947 / — |
| **ECE** (lower better) Jev / Laya / Terra | 0.110 / 0.520 / **0.081** | **0.200** / 0.317 / 0.303 | 0.042 / 0.496 / **0.020** | 0.022 / **0.020** / — |
| **AUROC** of confidence Jev / Laya / Terra | **0.831** / 0.696 / 0.807 | 0.611 / **0.711** / 0.636 | 0.935 / 0.897 / **0.964** | **0.909** / 0.885 / — |
| **p50 latency** Jev / Laya / Terra | 0.20 s / **0.10 s** / 1.04 s | 0.19 s / **0.04 s** / 1.06 s | 0.20 s / **0.08 s** / 1.04 s | 0.18 s / **0.08 s** / — |
| **Cost per 1k** Jev / Laya / Terra | $0.040 / **$0** / $2.02 | $0.014 / **$0** / $0.64 | $0.021 / **$0** / $0.85 | $0.021 / **$0** / — |

- **Laya is 2–5x faster than Jev here and free**, but the two numbers are not the same quantity: Laya is local compute, Jev and Terra include the network round trip. Laya's latency scales with option count, not just question count: 0.04 s at 5 levels, 0.08 s at 2 options, 0.10 s at 77.
- **Laya's `noul` collapses on this data.** It returns 0.0 on 298 of 300 reviews and scores 0.507, chance on a balanced binary task, with 99% of answers at 0.99 confidence or above. The same model, same 300 reviews, asked as a 2-option `choice`, scores 0.947. Jev scores 0.970 and 0.967 on the two framings. The collapse is unchanged by dropping `criteria` or by raising the context budget.
- **Laya ships over-confident**, as its card states: raw ECE 0.32 to 0.52 here, against the 0.466 it reports pre-temperature. Its published 0.081 is after fitting a temperature per question type and option count; these numbers are as-shipped.
- **High-cardinality choice is Jev's.** Raising `head_max_len` from 192 to 512, which Laya's card recommends for 50+ options, lifts intent from 0.370 to 0.463 and leaves the other tasks unchanged. Jev scores 0.780.
- **Where Laya works, its confidence is the less saturated.** On polarity as `choice`, 33% of its answers sit at 0.99 or above against Jev's 89%, at the same ECE. On 5-level sentiment its AUROC beats Jev's, 0.711 against 0.611: it ranks its own errors better while being 26 points less accurate.

Full tables, confidence distributions and reliability diagrams: [`results/summary.md`](results/summary.md).

![confidence](results/confidence.png)

## Run

```bash
cp .env.example .env        # TYPESAFE_API_KEY, OPENROUTER_API_KEY
uv sync
uv run python -m evaljev.run --model jev
uv run python -m evaljev.run --model openai/gpt-5.6-terra
uv run python -m evaljev.run --model openai/gpt-5.6-terra --reasoning medium
uv run python -m evaljev.run --model jev --tag rerun     # determinism check
uv run python -m evaljev.report
uv run pytest
```

Local checkpoints need the extra, and one call per task with `--model laya`:

```bash
uv sync --extra laya
HF_HUB_DISABLE_XET=1 uv run python -m evaljev.run --model laya
HF_HUB_DISABLE_XET=1 uv run python -m evaljev.run --model "laya@head=512,len=1024"
```

`--model` is `jev`, any OpenRouter model id, or `laya[:subfolder][@key=value,...]` where the keys override the checkpoint's config. `--reasoning` sets effort where the model supports it. `--cap` stops at that many dollars of OpenRouter spend; local models are free and never capped. Runs resume; delete `results/<task>/<model>.jsonl` to redo a pair.

A task is up to three files in `data/`:

```
<name>.jsonl         {"id", "text", "label"} per line
<name>.labels.json   the label list; for noul [no, yes]; optional for choice
<name>.task.json     {"kind": "choice"|"score"|"noul", "instructions": ..., "criteria": ... (noul),
                      "data": "<other task>" to reuse its items instead of copying them}
```

## Notes

- Metrics are in `evaljev/metrics.py` and unit-tested in `tests/`. Confidence is the probability of the chosen label: the returned distribution for Jev and Laya, the stated number for Terra.
- Latency is the wall-clock time of one successful call, same clock for every model; for local checkpoints that is the forward pass with the model already resident, measured after warm-up.
- Cost is `input_tokens × $0.042/M` for Jev (output free), OpenRouter's billed `usage.cost` for API models, and zero for local ones.
- Local models run one call at a time so the latency figure is uncontended; API models run 8 or 16 in flight.
- Laya's checkpoint is 808 MB. The Hugging Face xet transfer path stalled at zero bytes on this machine; `HF_HUB_DISABLE_XET=1` uses the classic path.
- Raw per-item records with request ids are in `results/<task>/<model>.jsonl`.
- Caveats: the datasets are public and old; Terra's confidence is self-reported and clusters at 0.98 to 0.99; one run from one machine on one day; no prompt tuning; at n = 300 differences under about 5 points are noise; the Banking77 mirror has 3,076 test rows against 3,080 in the original.
