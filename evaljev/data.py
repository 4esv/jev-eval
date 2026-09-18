"""Download public datasets and write seeded, stratified samples to data/<task>.jsonl."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 20260917
N = 300
DATA = Path(__file__).resolve().parent.parent / "data"

# NOTE: kind maps to the Jev question type used for the task.
TASKS = {
    # NOTE: mteb mirror of PolyAI/banking77; the original is a script dataset datasets>=4 cannot load.
    "intent": {"hf": "mteb/banking77", "split": "test", "kind": "choice"},
    "sentiment5": {"hf": "SetFit/sst5", "split": "test", "kind": "score"},
    "polarity": {"hf": "stanfordnlp/imdb", "split": "test", "kind": "noul", "max_words": 300},
}

SST5_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _rows(task: str) -> tuple[list[dict], list[str]]:
    spec = TASKS[task]
    ds = load_dataset(spec["hf"], split=spec["split"])
    if task == "intent":
        names = sorted(set(ds["label_text"]))
        rows = [{"text": r["text"], "label": r["label_text"]} for r in ds]
    elif task == "sentiment5":
        names = SST5_LABELS
        rows = [{"text": r["text"], "label": names[r["label"]]} for r in ds]
    else:
        names = ["negative", "positive"]
        rows = [
            {"text": r["text"].replace("<br />", " "), "label": names[r["label"]]}
            for r in ds
            if len(r["text"].split()) <= spec["max_words"]
        ]
    return rows, names


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin across labels so every class is represented as evenly as n allows."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    labels = sorted(by_label)
    rng.shuffle(labels)
    out, i = [], 0
    while len(out) < n and any(by_label.values()):
        bucket = by_label[labels[i % len(labels)]]
        if bucket:
            out.append(bucket.pop())
        i += 1
    rng.shuffle(out)
    return out


def build(task: str, n: int = N) -> Path:
    rows, names = _rows(task)
    sample = stratified(rows, n, SEED)
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{task}.jsonl"
    with path.open("w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({"id": f"{task}-{i:03d}", **r}) + "\n")
    (DATA / f"{task}.labels.json").write_text(json.dumps(names, indent=1) + "\n")
    return path


def load(task: str) -> tuple[list[dict], list[str]]:
    rows = [json.loads(l) for l in (DATA / f"{task}.jsonl").read_text().splitlines()]
    names = json.loads((DATA / f"{task}.labels.json").read_text())
    return rows, names


def stats(task: str) -> str:
    rows, names = load(task)
    counts = Counter(r["label"] for r in rows)
    # NOTE: chars/4 is a rough token estimate; real counts come from API usage fields.
    toks = sorted(len(r["text"]) / 4 for r in rows)
    return (
        f"{task}: n={len(rows)} classes={len(counts)}/{len(names)} "
        f"per-class min/max={min(counts.values())}/{max(counts.values())} "
        f"est tokens p50={toks[len(toks) // 2]:.0f} p95={toks[int(len(toks) * 0.95)]:.0f} max={toks[-1]:.0f}"
    )


if __name__ == "__main__":
    for t in TASKS:
        build(t)
        print(stats(t))
