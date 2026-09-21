"""One call per item to Jev (TypeSafe), any OpenRouter model, or a local Laya checkpoint.

API runners time the successful HTTP round trip; the Laya runner times the forward pass. Both use the
same clock, but they are not the same quantity: one includes the network, the other is local compute.
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

LAYA_REPO = "convaiinnovations/laya"
_LAYA_CFG = {"head": "head_max_len", "len": "max_len"}  # short names accepted in a model spec
_laya_agents: dict[str, object] = {}

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


async def jev(client: httpx.AsyncClient, task: str, names: list[str], item: dict,
              url: str = JEV_URL, model: str = JEV_MODEL, priced: bool = True) -> dict:
    """TypeSafe's System One contract. Any server speaking it (e.g. a local Kev) works by passing `url`."""
    body = {"state": item["text"], "model": model, "questions": {"q": jev_question(task, names)}}
    headers = {"Authorization": f"Bearer {_key('TYPESAFE_API_KEY', 'API_KEY')}"} if priced else {}
    data, latency, attempts, rid = await _post(client, url, headers, body)
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
        "cost_usd": (usage.get("input_tokens") or 0) * JEV_USD_PER_INPUT_TOKEN if priced else 0.0,
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


def laya_agent(spec: str):
    """Load and cache a Laya checkpoint. Spec: laya | laya:multilingual | laya@head=512,len=1024."""
    if spec not in _laya_agents:
        import laya as laya_pkg  # optional extra: uv sync --extra laya

        base, _, overrides = spec.partition("@")
        _, _, subfolder = base.partition(":")
        agent = laya_pkg.load(LAYA_REPO, **({"subfolder": subfolder} if subfolder else {}))
        for kv in filter(None, overrides.split(",")):
            k, v = kv.split("=")
            agent.cfg[_LAYA_CFG.get(k, k)] = int(v)
        _laya_agents[spec] = agent
    return _laya_agents[spec]


async def laya(client: httpx.AsyncClient, task: str, names: list[str], item: dict, spec: str) -> dict:
    """Local forward pass. Same question schema as Jev, so jev_question/jev_parse are reused."""
    agent = laya_agent(spec)
    body = {"q": jev_question(task, names)}
    t0 = time.perf_counter()
    data = await asyncio.to_thread(agent.predict, item["text"], body)
    latency = time.perf_counter() - t0
    ans = data["answers"]["q"]
    pred, probs, conf = jev_parse(task, names, ans)
    usage = data.get("usage", {})
    return {
        "model_version": spec,
        "pred": pred,
        "probs": probs,
        "conf": max(probs.values()),
        "ts_confidence": ans.get("confidence"),
        "latency_s": latency,
        "attempts": 1,
        "in_tok": usage.get("input_tokens"),
        "out_tok": usage.get("output_tokens"),
        "cost_usd": 0.0,  # self-hosted
        "request_id": "",
        "raw": ans,
    }


OPENJEV_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
_openjev: dict[str, object] = {}


def openjev_model(repo: str):
    """Load and cache open-jev. Its inference code ships inside the HF repo, so add the snapshot to sys.path."""
    if repo not in _openjev:
        import sys

        from huggingface_hub import snapshot_download

        path = snapshot_download(repo)
        if path not in sys.path:
            sys.path.insert(0, path)
        from typed_decisions.open_jev import OpenJev

        _openjev[repo] = OpenJev.from_pretrained(repo)
    return _openjev[repo]


async def openjev(client: httpx.AsyncClient, task: str, names: list[str], item: dict, spec: str) -> dict:
    """Local forward pass. Takes a list of questions with `options`; score probabilities are keyed by
    level name rather than index, and noul carries no confidence, so it needs its own parse."""
    _, _, repo = spec.partition(":")
    model = openjev_model(repo or OPENJEV_REPO)
    kind = task_spec(task)["kind"]
    q = {"type": kind, "instructions": task_spec(task)["instructions"]}
    if kind == "choice":
        q["options"] = [display(n) for n in names]
    elif kind == "score":
        q["options"] = names
    t0 = time.perf_counter()
    out = await asyncio.to_thread(model.decide, item["text"], [q])
    latency = time.perf_counter() - t0
    ans = out[0]
    if kind == "choice":
        back = {display(n): n for n in names}
        probs = {back[k]: v for k, v in ans["probabilities"].items()}
        pred = back[ans["choice"]]
    elif kind == "score":
        probs = dict(ans["probabilities"])
        pred = max(probs, key=probs.get)
    else:
        p = ans["noul"]
        probs = {names[1]: p, names[0]: 1 - p}
        pred = max(probs, key=probs.get)
    return {
        "model_version": spec,
        "pred": pred,
        "probs": probs,
        "conf": max(probs.values()),
        "ts_confidence": ans.get("confidence"),
        "latency_s": latency,
        "attempts": 1,
        "in_tok": None,
        "out_tok": None,
        "cost_usd": 0.0,
        "request_id": "",
        "raw": ans,
    }


def runner(model: str, effort: str | None):
    """'jev' -> TypeSafe; 'laya...' -> a local checkpoint; anything else -> that OpenRouter model id."""
    if model == "jev":
        return jev
    if model == "laya" or model.startswith(("laya:", "laya@")):
        async def run_laya(client, task, names, item):
            return await laya(client, task, names, item, model)
        return run_laya
    if model.startswith("kev"):
        # A local server speaking the System One contract; KEV_URL overrides the default port.
        url = os.environ.get("KEV_URL", "http://127.0.0.1:8009") + "/v1/systemone"
        _, _, name = model.partition(":")

        async def run_kev(client, task, names, item):
            return await jev(client, task, names, item, url=url, model=name or "kev", priced=False)
        return run_kev
    if model == "open-jev" or model.startswith("open-jev:"):
        async def run_openjev(client, task, names, item):
            return await openjev(client, task, names, item, model)
        return run_openjev

    async def run(client, task, names, item):
        return await openrouter(client, task, names, item, model, effort)
    return run


def slug(model: str, effort: str | None) -> str:
    """Filename-safe name for a model spec: openai/x -> openai__x, laya@head=512,len=1024 -> laya-head512-len1024."""
    s = model.replace("/", "__").replace(":", "-").replace("@", "-")
    for ch in ("=", ","):
        s = s.replace(ch, "-" if ch == "," else "")
    return s + (f"+{effort}" if effort else "")
