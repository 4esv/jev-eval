"""One call per item to Jev (TypeSafe) or any OpenRouter model, normalized to one record shape.

Both runners time only the successful HTTP round trip with the same clock, so latency is comparable.
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_USD_PER_INPUT_TOKEN = 0.042 / 1e6  # output tokens are free

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

def task_spec(task: str) -> dict:
    """data/<task>.task.json: {"kind": "choice"|"score"|"noul", "instructions": "...", "criteria": {...} (noul only)}."""
    return json.loads((Path(__file__).resolve().parent.parent / "data" / f"{task}.task.json").read_text())

RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


def load_env(path: Path = Path(__file__).resolve().parent.parent / ".env") -> None:
    """Minimal .env loader; real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _key(*names: str) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    raise SystemExit(f"missing API key: set one of {', '.join(names)}")


def display(label: str) -> str:
    return label.replace("_", " ")


async def _post(client: httpx.AsyncClient, url: str, headers: dict, body: dict) -> tuple[dict, float, int, str]:
    """POST with backoff on retryable statuses. Returns (json, latency_s, attempts, request_id)."""
    for attempt in range(1, 7):
        t0 = time.perf_counter()
        try:
            r = await client.post(url, headers=headers, json=body)
        except httpx.TransportError:
            if attempt == 6:
                raise
            await asyncio.sleep(min(30, 2**attempt) + random.random())
            continue
        latency = time.perf_counter() - t0
        if r.status_code in RETRY_STATUS and attempt < 6:
            wait = float(r.headers.get("retry-after") or min(30, 2**attempt))
            await asyncio.sleep(wait + random.random())
            continue
        r.raise_for_status()
        rid = r.headers.get("x-typesafe-request-id") or r.headers.get("x-generation-id") or ""
        return r.json(), latency, attempt, rid
    raise RuntimeError("unreachable")


def jev_question(task: str, names: list[str]) -> dict:
    spec = task_spec(task)
    if spec["kind"] == "choice":
        return {"type": "choice", "instructions": spec["instructions"], "criteria": {display(n): None for n in names}}
    if spec["kind"] == "score":
        return {"type": "score", "instructions": spec["instructions"], "criteria": names}
    return {"type": "noul", "instructions": spec["instructions"], "criteria": spec["criteria"]}


def jev_parse(task: str, names: list[str], ans: dict) -> tuple[str, dict, float | None]:
    """Returns (pred label, probs over label names, TypeSafe's own confidence field)."""
    kind = task_spec(task)["kind"]
    if kind == "choice":
        back = {display(n): n for n in names}
        probs = {back[k]: v for k, v in ans["probabilities"].items()}
        return back[ans["choice"]], probs, ans.get("confidence")
    if kind == "score":
        probs = {names[int(k)]: v for k, v in ans["probabilities"].items()}
        return max(probs, key=probs.get), probs, ans.get("confidence")
    p = ans["noul"]  # noul tasks: names[1] is the "yes" label
    probs = {names[1]: p, names[0]: 1 - p}
    return max(probs, key=probs.get), probs, None


async def jev(client: httpx.AsyncClient, task: str, names: list[str], item: dict) -> dict:
    body = {"state": item["text"], "model": JEV_MODEL, "questions": {"q": jev_question(task, names)}}
    headers = {"Authorization": f"Bearer {_key('TYPESAFE_API_KEY', 'API_KEY')}"}
    data, latency, attempts, rid = await _post(client, JEV_URL, headers, body)
    pred, probs, ts_conf = jev_parse(task, names, data["answers"]["q"])
    usage = data.get("usage", {})
    return {
        "model_version": data.get("model"),
        "pred": pred,
        "probs": probs,
        "conf": max(probs.values()),
        "ts_confidence": ts_conf,
        "latency_s": latency,
        "attempts": attempts,
        "in_tok": usage.get("input_tokens"),
        "out_tok": usage.get("output_tokens"),
        "cost_usd": (usage.get("input_tokens") or 0) * JEV_USD_PER_INPUT_TOKEN,
        "request_id": rid,
        "raw": data["answers"]["q"],
    }


def llm_messages(task: str, names: list[str], text: str) -> list[dict]:
    options = "\n".join(f"- {display(n)}" for n in names)
    system = (
        "You classify text for software. Answer with exactly one option from the list, and a "
        "confidence between 0 and 1 that your chosen option is correct. Be calibrated: of all "
        "answers you give with confidence 0.8, about 80% should be correct."
    )
    user = f"{task_spec(task)['instructions']}\n\nOptions:\n{options}\n\nText:\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def openrouter(client: httpx.AsyncClient, task: str, names: list[str], item: dict, model: str, effort: str | None = None) -> dict:
    """Any OpenRouter chat model with strict JSON-schema output; effort opts into reasoning where supported."""
    shown = [display(n) for n in names]
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": shown},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
        "additionalProperties": False,
    }
    body = {
        "model": model,
        "messages": llm_messages(task, names, item["text"]),
        "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": schema}},
        "usage": {"include": True},
    }
    if effort:
        body["reasoning"] = {"effort": effort}
    headers = {"Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}"}
    data, latency, attempts, rid = await _post(client, OR_URL, headers, body)
    content = data["choices"][0]["message"]["content"]
    out = json.loads(content)
    back = dict(zip(shown, names))
    pred, c = back[out["label"]], float(out["confidence"])
    # NOTE: the model states one confidence; other options get the remainder spread evenly.
    rest = (1 - c) / (len(names) - 1)
    probs = {n: (c if n == pred else rest) for n in names}
    usage = data.get("usage", {})
    cost = usage.get("cost")
    if cost is None:
        raise RuntimeError("OpenRouter omitted usage.cost; pass 'usage': {'include': True}")
    return {
        "model_version": data.get("model"),
        "provider": data.get("provider"),
        "pred": pred,
        "probs": probs,
        "conf": c,
        "ts_confidence": None,
        "latency_s": latency,
        "attempts": attempts,
        "in_tok": usage.get("prompt_tokens"),
        "out_tok": usage.get("completion_tokens"),
        "reasoning_tok": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost_usd": float(cost),
        "request_id": rid or data.get("id", ""),
        "raw": content,
    }


def runner(model: str, effort: str | None):
    """'jev' -> TypeSafe; anything else -> that OpenRouter model id."""
    if model == "jev":
        return jev

    async def run(client, task, names, item):
        return await openrouter(client, task, names, item, model, effort)
    return run


def slug(model: str, effort: str | None) -> str:
    return model.replace("/", "__") + (f"+{effort}" if effort else "")
