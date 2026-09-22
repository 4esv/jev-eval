"""Serve a local checkpoint under the System One contract, so any client that talks to TypeSafe can talk to it.

    uv run --extra serve python -m evaljev.serve --model laya --port 8010
    uv run --extra serve python -m evaljev.serve --model 'laya@head=512,len=1024' --port 8010

POST /v1/systemone takes {"state": ..., "questions": {...}} and returns {"answers": {...}, "usage": {...}}.
A state that is not a string is serialized to JSON, because the checkpoints take text. Laya's own predict()
accepts the same question schema, so the request passes through unchanged. GET /v1/models reports what is loaded.
"""

import argparse
import json
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .runners import laya_agent

app = FastAPI()
LOADED: dict = {}


class Request(BaseModel):
    state: object
    questions: dict
    model: str | None = None


def as_text(state: object) -> str:
    return state if isinstance(state, str) else json.dumps(state, separators=(",", ":"))


@app.get("/v1/models")
def models() -> dict:
    return {"models": [{"id": LOADED["spec"], "aliases": ["jev-latest"], "cfg": LOADED["cfg"]}]}


@app.post("/v1/systemone")
def systemone(req: Request) -> dict:
    t0 = time.perf_counter()
    try:
        out = LOADED["agent"].predict(as_text(req.state), req.questions)
    except Exception as e:  # a malformed question reaches the checkpoint as a KeyError or ValueError
        raise HTTPException(400, str(e))
    out["model"] = LOADED["spec"]
    out.setdefault("usage", {})
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="laya", help="laya | laya:<subfolder> | laya@head=512,len=1024")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    agent = laya_agent(a.model)
    LOADED.update(spec=a.model, agent=agent, cfg=dict(agent.cfg))
    uvicorn.run(app, host="127.0.0.1", port=a.port)
