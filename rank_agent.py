"""Planning agent: investigates the ranker's top-K candidates with read-only tools, then picks one.

The LLM only reads. Every tool is a Python function over local files or Jaeger queries; nothing
runs commands or touches Docker. Code re-validates the choice with policy.validate(), and on any
failure (no key, LLM error, invalid output) the ranker's #1 is used instead.
"""
import json
import os
import re
import statistics
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

import planner
import ranker
from discovery import CLIENT, COMPOSE_FILES, ENV_FILE, SERVER, _expand, _load_env, fetch_spans, load_flows, load_graph
from metrics import load_baseline
from policy import PROTECTED, blast_radius, validate

TOP_K = 3
MAX_TOOL_CALLS = 8
MAX_LLM_CALLS = MAX_TOOL_CALLS + 2   # a few turns can carry several tool calls; this bounds the loop
AGENT_LOG = Path("agent_log.jsonl")
SRC_ENV_VAR = "OTEL_DEMO_SRC"        # root of the otel-demo checkout (contains src/ and compose.yaml)
MAX_RESULT_CHARS = 3000              # tool results are cut to this before going back to the LLM
MAX_MATCHES = 20
MAX_LINE_CHARS = 160
MAX_PATTERN_CHARS = 200
MAX_FILE_BYTES = 1_000_000
SKIP_DIRS = {".git", "node_modules", "genproto", "vendor", "target", "bin", "obj", "dist", ".next",
             "__pycache__", "test", "tests", "cypress"}
SOURCE_EXT = {".go", ".py", ".js", ".mjs", ".ts", ".tsx", ".cs", ".rs", ".java", ".kt", ".rb", ".php", ".ex",
              ".exs", ".erl", ".cpp", ".cc", ".h", ".hpp", ".lua", ".yaml", ".yml", ".json", ".toml",
              ".properties", ".ini", ".conf", ".env"}
TEST_FILE = re.compile(r"(_test\.go|\.(test|spec)\.[jt]sx?|Tests?\.cs)$|^test_.*\.py$")
RESILIENCE_KEYS =re.compile(r"timeout|retr|deadline|backoff|circuit|limit|pool|keepalive", re.I)
RETRY_TRACES = 40
RETRY_LOOKBACK = timedelta(minutes=30)
TIMEOUT_CLUSTER = 0.10   # errored client spans within +-10% of their median look like a fixed timeout
METRICS, DIRECTIONS = ("error_rate", "p95_ms"), ("up", "down", "flat")

SYSTEM_PROMPT = f"""You are the planning agent of a chaos-engineering tool for the OpenTelemetry Astronomy Shop
(microservices in Docker Compose). A deterministic ranker scored candidate experiments; you get its top few.
Investigate them with the read-only tools, then choose ONE. You never run anything: code validates your
choice and a human approves it.

Method:
1. For each candidate, find the edges INTO the target (get_edges) and check what protects its callers on those
   edges: retries, timeouts/deadlines, fallbacks, circuit breakers.
   - search_source(service, pattern) greps one service's source (regex, case-insensitive). Search the CALLER's
     code, e.g. "retry|backoff|timeout|deadline|WithTimeout|circuit", or the call itself (e.g. "Charge\\(").
   - get_service_env(service) shows compose env vars (timeouts, retry settings) and resource limits.
   - retry_evidence(caller, callee) counts repeated client calls in recent traces and spots a fixed client timeout.
   - get_candidate, get_baseline and get_history give the ranker's features, baseline metrics and past runs.
2. Adjust each candidate's risk (0..1). Lower it when callers already retry with a timeout or degrade gracefully;
   raise it when a synchronous, user-facing call has neither. Tie every adjustment to evidence.
3. Budget: at most {MAX_TOOL_CALLS} tool calls in total, so ask the most decisive questions first.
4. Then reply with JSON only (no prose, no code fences), exactly this shape:
{{"ranking": [{{"candidate_id": str, "adjusted_risk": number, "adjusted_priority": number,
               "evidence": [{{"tool": str, "finding": str}}], "reason": str}}],
 "chosen": candidate_id,
 "hypothesis": str,
 "predictions": [{{"svc": str, "metric": "error_rate"|"p95_ms", "direction": "up"|"down"|"flat",
                  "min_factor": number}}]}}
Rules: one ranking entry per candidate; chosen must be one of the candidate ids. predictions name the target or
services in its blast radius; min_factor is the expected peak/baseline ratio (>= 1 when direction is up).
network_delay slows calls (p95 up; errors only where a caller times out); container_stop makes calls fail
(error_rate up) until the container is back."""

TOOL_DECLS = [
    {"name": "get_candidate", "description": "Ranker features, scores and why-line of one candidate.",
     "parameters": {"type": "object", "properties": {"id": {"type": "string", "description": "candidate id"}},
                    "required": ["id"]}},
    {"name": "get_edges", "description": "Callers and callees of a service with sync/async type and operations.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
    {"name": "get_baseline", "description": "Baseline request rate, error rate and latency of a service.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
    {"name": "get_history", "description": "Past chaos experiments on a service and their outcomes.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
    {"name": "search_source",
     "description": "Regex search (case-insensitive) over one service's source code. Max 20 matches as file:line.",
     "parameters": {"type": "object", "properties": {
         "service": {"type": "string", "description": "service directory name, e.g. checkout"},
         "pattern": {"type": "string", "description": "regex, e.g. retry|timeout|deadline"}},
         "required": ["service", "pattern"]}},
    {"name": "get_service_env",
     "description": "Environment variables, resource limits and restart policy of a service from the compose file.",
     "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}},
    {"name": "retry_evidence",
     "description": "From recent traces: does the caller repeat client calls to the callee (retries), and do its "
                    "failed calls end at a fixed duration (client timeout)?",
     "parameters": {"type": "object", "properties": {"caller": {"type": "string"}, "callee": {"type": "string"}},
                    "required": ["caller", "callee"]}},
]


class ToolError(Exception):
    pass


# ---- read-only tools -------------------------------------------------------------------

def _src_root():
    load_dotenv()
    raw = os.getenv(SRC_ENV_VAR, "").strip()
    if not raw:
        raise ToolError(f"{SRC_ENV_VAR} is not set")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise ToolError(f"{SRC_ENV_VAR}={raw} is not a directory")
    return root


def _service_dir(root, service):
    """root/src/<service>, refusing anything that is not a plain name or that resolves outside root."""
    if not isinstance(service, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", service) or ".." in service:
        raise ToolError(f"refused: {service!r} is not a service name")
    path = (root / "src" / service).resolve()
    if not path.is_relative_to(root):
        raise ToolError(f"refused: {service!r} resolves outside {SRC_ENV_VAR}")
    if not path.is_dir():
        raise ToolError(f"no source directory for {service!r}")
    return path


def search_source(service, pattern):
    """Regex search over $OTEL_DEMO_SRC/src/<service>; never follows symlinks out of the root."""
    root = _src_root()
    base = _service_dir(root, service)
    if not isinstance(pattern, str) or not pattern or len(pattern) > MAX_PATTERN_CHARS:
        raise ToolError(f"pattern must be 1-{MAX_PATTERN_CHARS} characters")
    if re.search(r"\([^)]*[+*][^)]*\)[+*{]", pattern):
        raise ToolError("refused: nested quantifiers (catastrophic backtracking)")
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        raise ToolError(f"bad regex: {e}")
    matches, files = [], 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d)))
        for name in sorted(filenames):
            p = Path(dirpath, name)
            if (p.suffix.lower() not in SOURCE_EXT or TEST_FILE.search(name) or p.is_symlink()
                    or not p.resolve().is_relative_to(root)):
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            files += 1
            for i, line in enumerate(lines, 1):
                if rx.search(line[:500]):
                    matches.append({"at": f"{p.relative_to(root).as_posix()}:{i}", "line": line.strip()[:MAX_LINE_CHARS]})
                    if len(matches) >= MAX_MATCHES:
                        return {"service": service, "pattern": pattern, "matches": matches, "truncated": True}
    return {"service": service, "pattern": pattern, "files_searched": files, "matches": matches, "truncated": False}


def get_service_env(service):
    """Compose config of one service: env vars (resilience-looking ones first), limits, restart policy."""
    root = _src_root()
    if not isinstance(service, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", service):
        raise ToolError(f"refused: {service!r} is not a service name")
    env = _load_env(root / ENV_FILE) if (root / ENV_FILE).is_file() else {}
    for fname in COMPOSE_FILES:
        path = root / fname
        if not path.is_file():
            continue
        svc = (yaml.safe_load(path.read_text(encoding="utf-8")).get("services") or {}).get(service)
        if svc is None:
            continue
        raw = svc.get("environment") or []
        pairs = raw.items() if isinstance(raw, dict) else (x.split("=", 1) if "=" in x else (x, None) for x in raw)
        values = {k: _expand(str(env.get(k, "") if v is None else v), env) for k, v in pairs}
        return {"service": service, "file": fname,
                "resilience_settings": {k: v for k, v in values.items() if RESILIENCE_KEYS.search(k)},
                "environment": {k: v[:120] for k, v in values.items() if not k.startswith("OTEL_")},
                "resource_limits": ((svc.get("deploy") or {}).get("resources") or {}).get("limits"),
                "restart": svc.get("restart"), "healthcheck": "healthcheck" in svc,
                "depends_on": sorted(svc.get("depends_on") or [])}
    raise ToolError(f"{service!r} not found in {', '.join(COMPOSE_FILES)}")


def retry_evidence(caller, callee, spans=None):
    """Repeated client calls caller -> callee under one parent span (retries), and a fixed client timeout."""
    if spans is None:
        spans = fetch_spans(limits={caller: RETRY_TRACES}, lookback=RETRY_LOOKBACK)
    children = defaultdict(list)
    for (tid, _), s in spans.items():
        children[(tid, s["parent"])].append(s)
    calls = []
    for (tid, sid), s in spans.items():
        if s["svc"] != caller or s["kind"] != CLIENT:
            continue
        a = s["attrs"]
        peer = str(a.get("server.address") or a.get("peer.service") or a.get("net.peer.name") or "").split(":")[0]
        if peer == callee or any(k["svc"] == callee and k["kind"] == SERVER for k in children[(tid, sid)]):
            calls.append(s)
    groups = defaultdict(list)
    for s in calls:
        groups[(s["trace"], s["parent"], s["attrs"].get("rpc.method") or s["name"])].append(s)
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    after_error = sum(any(x["status"] == 2 for x in sorted(v, key=lambda x: x["start"])[:-1]) for v in repeated.values())
    durs = sorted(s["dur"] for s in calls)
    errs = [s["dur"] for s in calls if s["status"] == 2]
    timeout = None
    if len(errs) >= 3:
        med = statistics.median(errs)
        if med > 0 and sum(abs(d - med) <= TIMEOUT_CLUSTER * med for d in errs) >= 0.6 * len(errs):
            timeout = round(med)
    return {"caller": caller, "callee": callee, "lookback_min": int(RETRY_LOOKBACK.total_seconds() // 60),
            "traces_checked": len({s["trace"] for s in spans.values() if s["svc"] == caller}),
            "client_calls": len(calls), "errored_calls": len(errs),
            "traces_with_repeated_calls": len({k[0] for k in repeated}),
            "max_repeats": max((len(v) for v in groups.values()), default=0),
            "repeats_after_error": after_error,
            "p50_ms": round(statistics.median(durs), 1) if durs else None,
            "max_ms": round(durs[-1], 1) if durs else None,
            "client_timeout_ms": timeout,
            "note": ("no errored calls in the window, so retries could not show up here; check the source"
                     if not errs else "repeats can be normal fan-out (one call per item); "
                                      "a retry usually follows an errored call")}


class Toolbox:
    """Dispatches the LLM's tool calls to the read-only functions above. Never raises."""

    def __init__(self, g, baseline, history, ranking):
        self.g, self.baseline, self.history = g, baseline or {}, history
        self.by_id = {c["id"]: c for c in ranking}

    def _service(self, service):
        if service not in self.g:
            raise ToolError(f"unknown service {service!r}")
        return service

    def get_candidate(self, id):
        c = self.by_id.get(id)
        if c is None:
            raise ToolError(f"unknown candidate {id!r}")
        return {k: c[k] for k in ("id", "rank", "risk", "info", "priority", "why")} | {
            "features": {k: v["raw"] for k, v in c["features"].items()}}

    def get_edges(self, service):
        s = self._service(service)
        edge = lambda d: {"mode": ranker._mode(d), "type": d.get("type"), "operations": (d.get("operations") or [])[:5],
                          "avg_latency_ms": d.get("avg_latency_ms"), "calls": d.get("call_count")}
        return {"service": s, "callers": {u: edge(d) for u, _, d in self.g.in_edges(s, data=True)},
                "callees": {v: edge(d) for _, v, d in self.g.out_edges(s, data=True)},
                "blast_radius": blast_radius(self.g, s)}

    def get_baseline(self, service):
        s = self._service(service)
        m = self.baseline.get("services", {}).get(s)
        if not m:
            return {"service": s, "note": "no span metrics (not instrumented, e.g. kafka or a database)"}
        r = lambda x: None if x is None else round(x, 2)
        return {"service": s, "requests": self.baseline.get("requests", {}).get(s),
                **{k: v and {f: r(v[f]) for f in ("mean", "std", "max")} for k, v in m.items()},
                "abort_thresholds": self.baseline.get("thresholds", {}).get("services", {}).get(s)}

    def get_history(self, service):
        runs = [r for r in self.history if r["target"] == self._service(service)]
        return {"service": service, "runs": len(runs), "latest": runs[-10:]}

    def search_source(self, service, pattern):
        return search_source(service, pattern)

    def get_service_env(self, service):
        return get_service_env(service)

    def retry_evidence(self, caller, callee):
        return retry_evidence(self._service(caller), self._service(callee))

    def call(self, name, args):
        fn = getattr(self, name, None) if name in {d["name"] for d in TOOL_DECLS} else None
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        try:
            return fn(**(args or {}))
        except ToolError as e:
            return {"error": str(e)}
        except Exception as e:  # e.g. Jaeger down, wrong arguments
            return {"error": f"{type(e).__name__}: {e}"[:300]}


def _compact(result):
    text = json.dumps(result, default=str)
    return result if len(text) <= MAX_RESULT_CHARS else {"truncated": True, "text": text[:MAX_RESULT_CHARS]}


# ---- log -------------------------------------------------------------------------------

class AgentLog:
    """Appends one JSON line per step to agent_log.jsonl so the UI can replay the reasoning."""

    def __init__(self, path=AGENT_LOG):
        self.path, self.run_id, self.step = Path(path), uuid.uuid4().hex[:8], 0

    def __call__(self, type_, summary, tool=None):
        self.step += 1
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": self.run_id,
               "step": self.step, "type": type_, "tool": tool, "summary": " ".join(str(summary).split())[:240]}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass  # logging must never break planning


# ---- the agent -------------------------------------------------------------------------

def _brief(top, g):
    """First user message: the candidates and the edges into each target."""
    cands = []
    for c in top:
        t = c["target"]
        cands.append({"id": c["id"], "risk": c["risk"], "info": c["info"], "priority": c["priority"], "why": c["why"],
                      "callers": {u: ranker._mode(d) + (f" {d['type']}" if d.get("type") else "")
                                  for u, _, d in g.in_edges(t, data=True)},
                      "blast_radius": blast_radius(g, t)})
    return ("Candidates (ranker scores are 0..1, higher priority = run first):\n" + json.dumps(cands, indent=1)
            + f"\nProtected services (never targets): {sorted(PROTECTED)}")


def _investigate(api_key, top, g, toolbox, log, post):
    """Bounded tool-calling loop. Returns (final text, model, tool calls used)."""
    contents = [{"role": "user", "parts": [{"text": _brief(top, g)}]}]
    used, model = 0, None
    for _ in range(MAX_LLM_CALLS):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents,
                "tools": [{"functionDeclarations": TOOL_DECLS}],
                "toolConfig": {"functionCallingConfig": {"mode": "AUTO" if used < MAX_TOOL_CALLS else "NONE"}},
                "generationConfig": {"maxOutputTokens": planner.LLM_MAX_TOKENS}}
        cand, model = post(api_key, body, (model,) if model else planner.LLM_MODELS)  # keep one model per conversation
        parts = (cand.get("content") or {}).get("parts") or []
        contents.append({"role": "model", "parts": parts})  # verbatim: Gemini needs its thought signatures back
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            return text, model, used
        if text.strip():
            log("thought", text)
        replies = []
        for fc in calls:
            name, args = fc.get("name"), fc.get("args") or {}
            if used >= MAX_TOOL_CALLS:
                result = {"error": "tool budget used up; give your final JSON answer now"}
            else:
                used += 1
                log("tool_call", f"{name}({json.dumps(args)})", tool=name)
                result = _compact(toolbox.call(name, args))
                log("tool_result", json.dumps(result, default=str), tool=name)
            replies.append({"functionResponse": {"name": name, "response": {"result": result},
                                                 **({"id": fc["id"]} if fc.get("id") else {})}})
        if used >= MAX_TOOL_CALLS:
            replies.append({"text": "Tool budget used up. Reply now with the final JSON only."})
        contents.append({"role": "user", "parts": replies})
    raise RuntimeError(f"no final answer within {MAX_LLM_CALLS} LLM calls")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_answer(obj, top, g):
    """Validate the LLM's JSON against the required shape. Returns it cleaned; raises ValueError."""
    ids = {c["id"] for c in top}
    if obj.get("chosen") not in ids:
        raise ValueError(f"chosen {obj.get('chosen')!r} is not one of the top-{len(top)} candidates")
    if not isinstance(obj.get("hypothesis"), str) or not obj["hypothesis"].strip():
        raise ValueError("hypothesis missing")
    entries = obj.get("ranking")
    if not isinstance(entries, list) or not entries:
        raise ValueError("ranking missing")
    ranking = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("candidate_id") not in ids:
            raise ValueError(f"ranking entry for unknown candidate: {str(e)[:80]}")
        if not all(_num(e.get(k)) for k in ("adjusted_risk", "adjusted_priority")):
            raise ValueError(f"{e['candidate_id']}: adjusted_risk/adjusted_priority must be numbers")
        ev = [{"tool": str(x.get("tool", ""))[:40], "finding": str(x.get("finding", ""))[:300]}
              for x in (e.get("evidence") or []) if isinstance(x, dict)]
        ranking[e["candidate_id"]] = {"adjusted_risk": round(min(1.0, max(0.0, e["adjusted_risk"])), 3),
                                      "adjusted_priority": round(e["adjusted_priority"], 3),
                                      "evidence": ev[:10], "reason": str(e.get("reason", ""))[:500]}
    preds = []
    for p in obj.get("predictions") or []:
        if not (isinstance(p, dict) and p.get("svc") in g and p.get("metric") in METRICS
                and p.get("direction") in DIRECTIONS and _num(p.get("min_factor"))):
            raise ValueError(f"bad prediction: {str(p)[:80]}")
        preds.append({k: p[k] for k in ("svc", "metric", "direction", "min_factor")})
    if not preds:
        raise ValueError("predictions missing")
    return {"chosen": obj["chosen"], "hypothesis": obj["hypothesis"].strip(), "predictions": preds, "ranking": ranking}


def _default_story(c, g):
    """Deterministic hypothesis and predictions for the ranker fallback."""
    t, d = c["target"], c["duration_s"]
    callers = sorted(u for u, _, e in g.in_edges(t, data=True) if ranker._mode(e) == "sync")  # protected ones too
    who = ", ".join(callers) or "no direct sync callers"
    if c["fault"] == "container_stop":
        hyp = (f"Stopping {t} for {d}s makes calls to it fail: the error rate of its callers ({who}) rises, "
               f"and everything returns to baseline once {t} restarts.")
        preds = [{"svc": s, "metric": "error_rate", "direction": "up", "min_factor": 2} for s in callers]
    else:
        hyp = (f"Adding {c['magnitude']}ms network delay to {t} for {d}s raises p95 latency of {t}'s callers ({who}) "
               f"without a large error increase, and latency returns to baseline after rollback.")
        preds = [{"svc": s, "metric": "p95_ms", "direction": "up", "min_factor": 1.5} for s in callers]
    return hyp, preds or [{"svc": t, "metric": "error_rate", "direction": "flat", "min_factor": 1}]


def _plan(c, hypothesis, reasoning):
    return {"target": c["target"], "fault": c["fault"], "magnitude": c["magnitude"], "duration_s": c["duration_s"],
            "hypothesis": hypothesis, "reasoning": reasoning}


def _top_k_rows(top, answer=None):
    rows = []
    for c in top:
        adj = (answer or {}).get("ranking", {}).get(c["id"], {})
        rows.append({"candidate_id": c["id"], "rank": c["rank"], "risk": c["risk"], "info": c["info"],
                     "priority": c["priority"], "why": c["why"],
                     "adjusted_risk": adj.get("adjusted_risk"), "adjusted_priority": adj.get("adjusted_priority"),
                     "evidence": adj.get("evidence", []), "reason": adj.get("reason")})
    return rows


def top_k(ranking, k):
    """Best k candidates with distinct targets: the evidence is per edge into a target, so two
    variants of one service would spend the tool budget twice on the same question."""
    seen, out = set(), []
    for c in ranking:
        if c["target"] not in seen:
            seen.add(c["target"])
            out.append(c)
    return out[:k]


def run_agent(k=TOP_K, api_key=None, post=None, graph=None, baseline=None, flows=None, history=None,
              log_path=AGENT_LOG):
    """Rank, investigate the top-k, choose one. Raises only if ranking itself is impossible
    (no dependency map / baseline); every LLM problem falls back to the ranker's #1."""
    if graph is None:
        graph, _ = load_graph()
    baseline = load_baseline() if baseline is None else baseline
    flows = load_flows() if flows is None else flows
    history = ranker.load_history() if history is None else history
    ranking = ranker.rank(graph, baseline, flows, history)
    if not ranking:
        raise ValueError("ranker produced no valid candidates")
    top = top_k(ranking, k)
    log = AgentLog(log_path)
    log("thought", "investigating top-%d: %s" % (len(top), ", ".join(f"{c['id']} ({c['priority']})" for c in top)))

    def fallback(reason):
        best = top[0]
        hyp, preds = _default_story(best, graph)
        log("decision", f"ranker_fallback -> {best['id']}: {reason}")
        return {"source": "ranker_fallback", "note": reason, "model": None, "run_id": log.run_id, "tool_calls": 0,
                "chosen": best["id"], "predictions": preds, "top_k": _top_k_rows(top),
                "plan": _plan(best, hyp, f"Ranker #1 (priority {best['priority']}): {best['why']}. "
                                         f"LLM investigation not used: {reason}")}

    api_key = api_key or planner.load_api_key()
    if not api_key:
        return fallback(f"{planner.API_KEY_VAR} is not set")
    try:
        text, model, used = _investigate(api_key, top, graph, Toolbox(graph, baseline, history, ranking), log,
                                         post or planner._post_gemini)
        answer = _check_answer(planner._parse_json(text), top, graph)
    except Exception as e:  # LLM/network failure or invalid output
        return fallback(f"LLM investigation failed: {type(e).__name__}: {e}"[:300])
    chosen = next(c for c in top if c["id"] == answer["chosen"])
    reason = answer["ranking"].get(chosen["id"], {}).get("reason") or chosen["why"]
    plan = _plan(chosen, answer["hypothesis"], reason)
    check = validate(plan, graph)
    if not check["valid"]:
        return fallback("agent's choice failed policy: " + "; ".join(check["reasons"]))
    log("decision", f"chose {chosen['id']} after {used} tool calls: {reason}")
    return {"source": "rank_agent", "note": f"model {model}, {used} tool calls", "model": model,
            "run_id": log.run_id, "tool_calls": used, "chosen": chosen["id"], "predictions": answer["predictions"],
            "top_k": _top_k_rows(top, answer), "plan": plan}


if __name__ == "__main__":
    out = run_agent()
    print(json.dumps({k: out[k] for k in ("source", "note", "chosen", "plan", "predictions", "top_k")}, indent=2))
    sys.exit(0)
