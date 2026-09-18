"""Resumable runner: appends one JSON line per item to results/<task>/<model>[.<tag>].jsonl.

    uv run python -m evaljev.run --task intent --model jev --n 20
    uv run python -m evaljev.run --task all --model terra --n 300 --cap 10
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from evaljev.data import TASKS, load
from evaljev.runners import RUNNERS, load_env

RESULTS = Path(__file__).resolve().parent.parent / "results"
CONCURRENCY = {"jev": 16, "terra": 8}


def out_path(task: str, model: str, tag: str) -> Path:
    return RESULTS / task / (f"{model}.{tag}.jsonl" if tag else f"{model}.jsonl")


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def spent(model: str) -> float:
    """Total recorded spend for a model across every results file, all tasks and tags."""
    return sum(r.get("cost_usd") or 0 for p in RESULTS.glob(f"*/{model}*.jsonl") for r in read(p))


async def run(task: str, model: str, n: int, tag: str, cap: float) -> None:
    rows, names = load(task)
    path = out_path(task, model, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read(path) if not r.get("error")}
    todo = [r for r in rows[:n] if r["id"] not in done]
    total = spent(model)
    print(f"{task}/{model}{'.' + tag if tag else ''}: {len(done)} done, {len(todo)} to run, spent so far ${total:.4f}")
    if not todo:
        return

    sem = asyncio.Semaphore(CONCURRENCY[model])
    lock = asyncio.Lock()
    stop = asyncio.Event()
    fn = RUNNERS[model]

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        nonlocal total
        async with sem:
            if stop.is_set():
                return
            if total >= cap:
                stop.set()
                print(f"ABORT: {model} spend ${total:.4f} reached cap ${cap}")
                return
            rec = {"id": item["id"], "task": task, "model": model, "gold": item["label"], "ts": time.time()}
            try:
                rec |= await fn(client, task, names, item)
            except Exception as e:  # recorded, retried on the next resume
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            async with lock:
                total += rec.get("cost_usd") or 0
                with path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=15)) as client:
        await asyncio.gather(*(one(client, it) for it in todo))

    final = read(path)
    # NOTE: an item that failed then succeeded on resume keeps both lines; the latest wins in metrics.
    ok = {r["id"] for r in final if not r.get("error")}
    errs = [r for r in final if r.get("error") and r["id"] not in ok]
    print(f"  -> {len(ok)} ok, {len(errs)} unresolved errors, {model} spend now ${total:.4f}")
    for r in errs[:3]:
        print("    ", r["id"], r["error"][:200])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=[*TASKS, "all"])
    ap.add_argument("--model", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", default="", help="separate output file, e.g. 'rerun' for determinism")
    ap.add_argument("--cap", type=float, default=10.0, help="abort when this model's total spend (USD) reaches this")
    a = ap.parse_args()
    load_env()
    for t in TASKS if a.task == "all" else [a.task]:
        asyncio.run(run(t, a.model, a.n, a.tag, a.cap))


if __name__ == "__main__":
    main()
