"""Download public datasets and write seeded, stratified samples to data/<task>.jsonl."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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
    from datasets import load_dataset  # NOTE: only rebuilding the shipped samples needs it (--group build)

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


def tasks() -> list[str]:
    """Every task with a data/<task>.task.json sidecar."""
    return sorted(p.name[: -len(".task.json")] for p in DATA.glob("*.task.json"))


KINDS = ("choice", "score", "noul")


def load(task: str) -> tuple[list[dict], list[str]]:
    """Rows and label names, validated. labels.json is optional for `choice` (defaults to the labels in the data)."""
    spec_path, rows_path, names_path = (DATA / f"{task}{ext}" for ext in (".task.json", ".jsonl", ".labels.json"))
    for p in (spec_path, rows_path):
        if not p.exists():
            raise SystemExit(f"{task}: missing {p.relative_to(DATA.parent)}; tasks available: {', '.join(tasks())}")
    spec = json.loads(spec_path.read_text())
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    names = json.loads(names_path.read_text()) if names_path.exists() else None
    errors = check(spec, rows, names)
    if errors:
        raise SystemExit(f"{task}: invalid task\n  " + "\n  ".join(errors))
    return rows, names if names is not None else sorted({r["label"] for r in rows})


def check(spec: dict, rows: list[dict], names: list[str] | None) -> list[str]:
    """Every problem with a task definition, as messages; empty when it can be run."""
    errors = []
    kind = spec.get("kind")
    if kind not in KINDS:
        errors.append(f"task.json kind is {kind!r}; expected one of {', '.join(KINDS)}")
    if not spec.get("instructions"):
        errors.append("task.json needs instructions")
    if kind == "noul" and set(spec.get("criteria") or {}) != {"true", "false"}:
        errors.append('noul task.json needs criteria {"true": ..., "false": ...}')
    if names is None and kind in ("score", "noul"):
        errors.append(f"{kind} needs labels.json: label order carries meaning (score: low to high; noul: [no, yes])")
    if names is not None:
        if len(set(names)) != len(names):
            errors.append("labels.json has duplicates")
        if kind == "noul" and len(names) != 2:
            errors.append(f"noul needs exactly 2 labels [no, yes]; labels.json has {len(names)}")
    if not rows:
        errors.append("no rows")
    bad = [i for i, r in enumerate(rows, 1) if not all(isinstance(r.get(k), str) and r[k] for k in ("id", "text", "label"))]
    if bad:
        errors.append(f'{len(bad)} rows are not {{"id", "text", "label"}} strings; first at line {bad[0]}')
        return errors
    ids = Counter(r["id"] for r in rows)
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate ids, e.g. {dup[0]!r}")
    if names is not None:
        unknown = sorted({r["label"] for r in rows} - set(names))
        if unknown:
            errors.append(f"{len(unknown)} labels in the data are not in labels.json, e.g. {unknown[0]!r}")
    return errors


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
