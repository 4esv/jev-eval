"""Resumable runner: appends one JSON line per item to results/<task>/<model>[.<tag>].jsonl.

    uv run python -m evaljev.run --task intent --model jev --n 20
    uv run python -m evaljev.run --task all --model openai/gpt-5.6-terra --n 300 --cap 10
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from evaljev.data import load, tasks
from evaljev.runners import load_env, runner, slug

RESULTS = Path(__file__).resolve().parent.parent / "results"
CONCURRENCY = 16  # Jev; OpenRouter models use OR_CONCURRENCY, local models run one at a time
OR_CONCURRENCY = 8


def out_path(task: str, name: str, tag: str) -> Path:
    return RESULTS / task / (f"{name}~{tag}.jsonl" if tag else f"{name}.jsonl")  # ~ separates a tag; model slugs contain dots


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def spent(model: str) -> float:
    """Recorded spend across every results file. Jev is capped alone; OpenRouter models share one cap.
    Local models are free and excluded, so a past API run cannot abort them."""
    if model.startswith(("laya", "open-jev", "kev")):
        return 0.0
    files = RESULTS.glob("*/jev*.jsonl") if model == "jev" else (
        p for p in RESULTS.glob("*/*.jsonl") if not p.name.startswith(("jev", "laya", "open-jev", "kev")))
    return sum(r.get("cost_usd") or 0 for p in files for r in read(p))


async def run(task: str, model: str, effort: str | None, n: int, tag: str, cap: float) -> None:
    rows, names = load(task)
    name = slug(model, effort)
    path = out_path(task, name, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read(path) if not r.get("error")}
    todo = [r for r in rows[:n] if r["id"] not in done]
    total = spent(model)
    print(f"{task}/{name}{'~' + tag if tag else ''}: {len(done)} done, {len(todo)} to run, spent so far ${total:.4f}")
    if not todo:
        return

    local = model.startswith(("laya", "open-jev", "kev"))
    sem = asyncio.Semaphore(1 if local else CONCURRENCY if model == "jev" else OR_CONCURRENCY)
    lock = asyncio.Lock()
    stop = asyncio.Event()
    fn = runner(model, effort)

    async def one(client: httpx.AsyncClient, item: dict) -> None:
        nonlocal total
        async with sem:
            if stop.is_set():
                return
            if total >= cap:
                stop.set()
                print(f"ABORT: {model} spend ${total:.4f} reached cap ${cap}")
                return
            rec = {"id": item["id"], "task": task, "model": name, "gold": item["label"], "ts": time.time()}
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
    ap.add_argument("--task", default="all", help="a task name from data/, or all")
    ap.add_argument("--model", required=True, help="jev, any OpenRouter model id (e.g. openai/gpt-5.6-terra), laya[:subfolder][@key=value,...], open-jev, or kev")
    ap.add_argument("--reasoning", default=None, help="OpenRouter reasoning effort: low, medium, high")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tag", default="", help="separate output file, e.g. 'rerun' for determinism")
    ap.add_argument("--cap", type=float, default=10.0, help="abort when total spend (USD) reaches this; all OpenRouter models share it")
    a = ap.parse_args()
    load_env()
    for t in (tasks() if a.task == "all" else [a.task]):
        asyncio.run(run(t, a.model, a.reasoning, a.n, a.tag, a.cap))


if __name__ == "__main__":
    main()
