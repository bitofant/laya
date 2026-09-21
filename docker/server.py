"""HTTP inference endpoint for Laya.

Typed questions in, calibrated distributions out. The model is loaded once at startup
and held on the GPU; requests are pure forward passes.

Endpoints:
  GET  /health        liveness + device/VRAM (never loads the model)
  GET  /presets       names of the SDK's built-in question sets
  POST /decide        {"state": ..., "questions": {...}} or {"state": ..., "preset": "triage"}
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, model_validator

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/models/laya-typed-decisions")
DEVICE = os.environ.get("LAYA_DEVICE", "cuda")
# laya.Agent is not documented as thread-safe and holds per-call CUDA state, so requests
# are serialised through one lock. Throughput still comes from batching questions within
# a request (2.3 ms/question batched vs 6.2 ms standalone), not from concurrent calls.
import asyncio

STATE: Dict[str, Any] = {"agent": None, "lock": asyncio.Lock(), "presets": {}}

API_KEY = os.environ.get("LAYA_API_KEY")
auth_scheme = HTTPBearer(auto_error=False)

async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Security(auth_scheme)):
    if not API_KEY:
        return  # No API key configured, allow all
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    import laya

    # Fail fast at startup rather than on the first request: a container that cannot
    # serve should not report itself as up.
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"weights not found at {MODEL_DIR} -- run ./setup.sh")

    agent = laya.load(MODEL_DIR, device=DEVICE)
    # laya.load() degrades to CPU with only a printed warning; refuse to serve silently
    # at ~10x latency when the caller asked for cuda.
    if DEVICE == "cuda" and agent.device.type != "cuda":
        raise RuntimeError(
            "model fell back to CPU (usually VRAM exhaustion; see docs/gpu-notes.md). "
            "Set LAYA_DEVICE=cpu to serve on CPU deliberately."
        )
    # Warm up before serving. The first forward pass pays CUDA context setup, cuBLAS
    # handle creation and kernel autotune: measured at ~14.7 s on an RTX 5090, versus
    # ~6.5 ms steady state. Without this the first caller eats that, which reads as a
    # hang or a timeout. Each question *type* is warmed because they take different
    # paths through the decision head (a new shape cost ~189 ms even after the first pass).
    warm = {
        "w_choice": {
            "type": "choice",
            "instructions": "warmup",
            "criteria": {"a": "first option", "b": "second option"},
        },
        "w_score": {"type": "score", "instructions": "warmup", "criteria": ["lo", "mid", "hi"]},
        "w_noul": {"type": "noul", "instructions": "warmup"},
    }
    t0 = time.perf_counter()
    for _ in range(2):
        agent.system_one("warmup state for kernel autotuning", warm)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[laya] warmup done in {time.perf_counter() - t0:.1f}s")

    STATE["agent"] = agent
    STATE["presets"] = {
        "triage": laya.triage_questions,
        "email": laya.email_questions,
        "guard": laya.guard_questions,
        "moderation": laya.moderation_questions,
        "router": laya.router_questions,
    }
    print(f"[laya] ready on {agent.device} | {MODEL_DIR}")
    yield
    STATE["agent"] = None


app = FastAPI(title="laya", version="0.1.0", lifespan=lifespan)


class Question(BaseModel):
    type: str = Field(description="choice | score | noul")
    instructions: str
    # choice -> {option: description}; score -> [level, ...]; noul -> omitted
    criteria: Optional[Union[Dict[str, str], List[str]]] = None

    @model_validator(mode="after")
    def check_shape(self):
        if self.type not in ("choice", "score", "noul"):
            raise ValueError(f"unknown type {self.type!r}; expected choice, score or noul")
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("choice questions need criteria as {option: description}")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("score questions need criteria as an ordered list of levels")
        if self.type == "choice" and len(self.criteria or {}) < 2:
            raise ValueError("choice questions need at least 2 options")
        return self


class DecideRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Optional[Dict[str, Question]] = None
    preset: Optional[str] = None

    @model_validator(mode="after")
    def one_of(self):
        if bool(self.questions) == bool(self.preset):
            raise ValueError("provide exactly one of 'questions' or 'preset'")
        return self


@app.get("/health")
def health():
    agent = STATE["agent"]
    body: Dict[str, Any] = {"status": "ok" if agent else "loading", "model_dir": MODEL_DIR}
    if agent:
        body["device"] = str(agent.device)
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        body["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_free_gb": round(free_b / 1e9, 2),
            "vram_total_gb": round(total_b / 1e9, 2),
        }
    return JSONResponse(body, status_code=200 if agent else 503)


@app.get("/presets")
def presets(token: Any = Depends(get_api_key)):
    return {"presets": sorted(STATE["presets"])}


@app.post("/decide")
async def decide(req: DecideRequest, token: Any = Depends(get_api_key)):
    agent = STATE["agent"]
    if agent is None:
        raise HTTPException(503, "model still loading")

    if req.preset:
        fn = STATE["presets"].get(req.preset)
        if fn is None:
            raise HTTPException(
                404, f"unknown preset {req.preset!r}; available: {sorted(STATE['presets'])}"
            )
        questions = fn()
    else:
        questions = {
            qid: q.model_dump(exclude_none=True) for qid, q in (req.questions or {}).items()
        }

    t0 = time.perf_counter()
    try:
        async with STATE["lock"]:
            result = agent.system_one(req.state, questions)
    except ValueError as exc:
        # Raised e.g. when a question's options exceed head_max_len -- caller's fault.
        raise HTTPException(422, str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        raise HTTPException(503, f"GPU out of memory: {exc}") from exc

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result
