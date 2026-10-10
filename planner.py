"""Experiment planner: PROPOSES one experiment (ranker + investigating agent). It never runs anything."""
import json
import os
import re

import requests
from dotenv import load_dotenv

# Tried in order: Gemini models get overloaded or retired, so one name is a single point of failure.
LLM_MODELS = ("gemini-3.8-flash", "gemini-3-flash-preview", "gemini-flash-lite-latest")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LLM_MAX_TOKENS = 4096  # thinking models spend part of this on reasoning before the answer
LLM_TIMEOUT_S = 20
VERDICTS = ("held", "not held", "inconclusive")
API_KEY_VAR = "LLM_API_KEY"
PLACEHOLDER_KEY = "your-api-key-here"  # value shipped in .env.example
FALLBACK_PLAN = {
    "target": "cart", "fault": "container_stop", "magnitude": 0, "duration_s": 60,
    "hypothesis": "Stopping cart for 60s makes add-to-cart, view-cart and checkout fail visibly "
                  "(error rate rises on frontend and checkout) and the system recovers once cart restarts.",
    "reasoning": "Safe default: cart is on the checkout path and is called by both frontend and checkout, "
                 "so it exercises the core purchase journey, yet it is not a protected service.",
}


def _parse_json(text):
    """Strip code fences, fall back to the outermost {...}. Raises ValueError."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        obj = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("no JSON object in LLM reply")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM reply is not a JSON object")
    return obj


def load_api_key():
    """The Gemini API key from .env / the environment, or None if unset or still the placeholder."""
    load_dotenv()
    key = os.getenv(API_KEY_VAR, "").strip()
    return None if not key or key == PLACEHOLDER_KEY else key


def _post_gemini(api_key, body, models=LLM_MODELS):
    """POST one generateContent request, trying each model in turn -> (candidate, model)."""
    errors = []
    for model in models:  # next model on timeout, overload, or retired model
        try:
            resp = requests.post(GEMINI_URL.format(model=model), json=body, timeout=LLM_TIMEOUT_S,
                                 headers={"x-goog-api-key": api_key})
        except requests.RequestException as e:
            errors.append(f"{model}: {type(e).__name__}")
            continue
        if resp.ok:
            cand = resp.json()["candidates"][0]
            if cand.get("finishReason") == "MAX_TOKENS":
                errors.append(f"{model}: reply truncated (MAX_TOKENS)")
                continue
            return cand, model
        errors.append(f"{model}: HTTP {resp.status_code}")
    raise RuntimeError("all models failed (" + "; ".join(errors) + ")")


def _call_llm(api_key, content, system):
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": "Context:\n" + json.dumps(content, indent=1)}]}],
            "generationConfig": {"maxOutputTokens": LLM_MAX_TOKENS, "responseMimeType": "application/json"}}
    cand, model = _post_gemini(api_key, body)
    return "".join(p.get("text", "") for p in cand["content"]["parts"]), model


def propose_experiment():
    """Return (plan, source, note). source: "rank_agent" (LLM investigated the ranker's top-K),
    "ranker_fallback" (ranker #1, LLM unavailable or invalid) or "fallback" (FALLBACK_PLAN).
    The plan carries the top-K ranking and the agent's evidence under the extra "ranking" key."""
    from rank_agent import run_agent  # imported here: rank_agent imports this module
    try:
        out = run_agent()
    except Exception as e:  # no dependency map / baseline, or a bug: degrade to the safe plan
        return dict(FALLBACK_PLAN), "fallback", f"ranking failed: {type(e).__name__}: {e}"[:300]
    plan = dict(out["plan"])
    plan["ranking"] = {k: out[k] for k in ("source", "note", "model", "run_id", "tool_calls", "chosen",
                                           "predictions", "top_k")}
    return plan, out["source"], out["note"]


JUDGE_PROMPT = """You judge the outcome of a chaos engineering experiment.
Given the hypothesis and the measured results, decide whether the hypothesis held.
Reply with a single JSON object and nothing else (no prose, no code fences):
{"verdict": "held" | "not held" | "inconclusive", "explanation": str}
Use "inconclusive" when the data is too sparse or noisy to decide, or the experiment was aborted
before the hypothesis could be tested. Telemetry notes: error rate and p95 are computed from server
spans seen in the last few seconds; services with low traffic may show no data (null).
Units: every error_rate value is a PERCENTAGE (0.6 means 0.6%, not 60%); p95_ms is milliseconds;
"change" is peak minus baseline in the same unit. Quote numbers exactly as given."""


def judge_hypothesis(plan, results):
    """Ask the same LLM whether the hypothesis held -> {verdict, explanation}; inconclusive on any failure."""
    api_key = load_api_key()
    if not api_key:
        return {"verdict": "inconclusive", "explanation": f"LLM unavailable: {API_KEY_VAR} is not set"}
    try:
        content = {"hypothesis": plan["hypothesis"], "experiment": {k: plan[k] for k in ("target", "fault", "magnitude", "duration_s")},
                   "results": results}
        text, model = _call_llm(api_key, content, system=JUDGE_PROMPT)
        out = _parse_json(text)
        if out.get("verdict") not in VERDICTS:
            raise ValueError(f"bad verdict {out.get('verdict')!r}")
        return {"verdict": out["verdict"], "explanation": str(out.get("explanation", "")), "model": model}
    except Exception as e:
        return {"verdict": "inconclusive", "explanation": f"LLM unavailable: {type(e).__name__}: {e}"[:300]}
