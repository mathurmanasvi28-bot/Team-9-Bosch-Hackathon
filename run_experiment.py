"""Experiment flow: propose -> validate -> print -> approve -> inject -> guard -> rollback -> recover -> report."""
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import guard
import inject
from metrics import load_baseline
from planner import judge_hypothesis, propose_experiment
from policy import validate

APPROVED_JSON = Path("approved_experiment.json")
RESULTS_DIR = Path("results")
ROLLBACK_TRIES = 3
# Kill-switch proof: deliberately tiny thresholds, so the guard aborts within seconds of injection.
FORCE_ABORT_TEST = False
FORCE_THRESHOLDS = {"error_rate": 0.0, "p95_ms": 0.001}
SOURCES = {"rank_agent": "RANK AGENT (LLM investigated the ranker's top candidates)",
           "ranker_fallback": "RANKER #1 (LLM investigation not used)",
           "fallback": "FALLBACK PLAN (ranking unavailable)"}


def print_ranking(ranking):
    """Ranked top-K table, then the chosen candidate's evidence and predictions."""
    rows = ranking.get("top_k") or []
    print(f"\nTop-{len(rows)} candidates (ranker score -> agent-adjusted):")
    print(f"  {'':2}{'candidate':<34}{'risk':>6}{'info':>6}{'prio':>6}{'adj risk':>10}{'adj prio':>10}")
    adj = lambda x: "-" if x is None else f"{x:.2f}"
    for r in rows:
        mark = "->" if r["candidate_id"] == ranking.get("chosen") else ""
        print(f"  {mark:<2}{r['candidate_id']:<34}{r['risk']:>6.2f}{r['info']:>6.2f}{r['priority']:>6.2f}"
              f"{adj(r['adjusted_risk']):>10}{adj(r['adjusted_priority']):>10}")
        print(f"  {'':2}  why: {r['why']}")
    chosen = next((r for r in rows if r["candidate_id"] == ranking.get("chosen")), None)
    if chosen and chosen["evidence"]:
        print(f"\nEvidence for {chosen['candidate_id']}:")
        for e in chosen["evidence"]:
            print(f"  - [{e['tool']}] {e['finding']}")
    if ranking.get("predictions"):
        print("Predictions: " + "; ".join(f"{p['svc']} {p['metric']} {p['direction']} (x{p['min_factor']})"
                                          for p in ranking["predictions"]))


def print_summary(plan, source, note, result):
    line = "=" * 64
    print(f"{line}\nPROPOSED EXPERIMENT\n{line}")
    for k in ("target", "fault", "magnitude", "duration_s"):
        print(f"{k:>12}: {plan.get(k)}")
    print(f"{'hypothesis':>12}: {plan.get('hypothesis')}\n{'reasoning':>12}: {plan.get('reasoning')}")
    if plan.get("ranking"):
        print_ranking(plan["ranking"])
    print(f"\nSource: {SOURCES.get(source, source)} - {note}")
    print(f"\nValidation: {'PASSED' if result['valid'] else 'FAILED'}")
    for r in result["reasons"]:
        print(f"  - {r}")
    br = result["blast_radius"]
    print(f"Blast radius ({len(br)} services depend on {plan.get('target')}): {', '.join(br) or 'none'}\n{line}")


def _rollback():
    """Roll back with retries, immune to Ctrl+C. Never skipped, never silent about failure."""
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for attempt in range(1, ROLLBACK_TRIES + 1):
            print(f"Rolling back (attempt {attempt}/{ROLLBACK_TRIES})...", flush=True)
            if inject.rollback():
                print("Rollback verified.")
                return True
            time.sleep(3)
        print(f"!!! ROLLBACK FAILED. Fix manually using {inject.ACTIVE_FILE} "
              f"(container_stop: `docker start <container>`; network_delay: remove the netem qdisc).")
        return False
    finally:
        signal.signal(signal.SIGINT, old)


def _recover_crashed_run():
    """If a previous run died mid-experiment, undo its fault before doing anything else."""
    if not inject.ACTIVE_FILE.exists():
        return True
    try:
        active = json.loads(inject.ACTIVE_FILE.read_text(encoding="utf-8"))
        print(f"Found {inject.ACTIVE_FILE} from an unfinished run ({active['fault']} on {active['target']}, "
              f"started {active['started_at']}). Rolling it back before anything else.")
    except (OSError, ValueError, KeyError) as e:
        print(f"{inject.ACTIVE_FILE} exists but is unreadable ({e}); fix or delete it by hand.")
        return False
    return _rollback()


def _fmt(x, unit=""):
    return "-" if x is None else f"{x:.1f}{unit}"


def print_results(plan, g, rolled_back, recovery, impact, containers, verdict):
    line = "=" * 64
    print(f"\n{line}\nEXPERIMENT RESULT\n{line}")
    print(f"Experiment: {plan['fault']} on {plan['target']}, magnitude {plan['magnitude']}, planned {plan['duration_s']}s")
    print(f"Outcome:    {'ABORTED' if g['aborted'] else 'COMPLETED'} after {g['elapsed_s']}s - {g['reason']}")
    print(f"Rollback:   {'verified' if rolled_back else 'FAILED - see message above'}")
    print("\nImpact vs baseline (baseline -> peak, change):")
    print(f"  {'service':<16}{'error rate %':<28}{'p95 latency ms'}")
    for r in impact:
        e, p = r["error_rate"], r["p95_ms"]
        print(f"  {r['service']:<16}{_fmt(e['baseline']):>6} -> {_fmt(e['peak']):>6} ({_fmt(e['change']):>6})      "
              f"{_fmt(p['baseline']):>7} -> {_fmt(p['peak']):>7} ({_fmt(p['change']):>7})")
    if containers:
        print("\nContainer resources, baseline -> peak (reference only, not abort conditions):")
        for c in containers:
            cpu, mem = c["cpu_percent"], c["memory_mb"]
            print(f"  {c['container']:<16}cpu {_fmt(cpu['baseline'], '%'):>7} -> {_fmt(cpu['peak'], '%'):>7}    "
                  f"mem {_fmt(mem['baseline'], 'MB'):>8} -> {_fmt(mem['peak'], 'MB'):>8}")
    if recovery["recovered"]:
        print(f"\nRecovery:   {recovery['seconds']}s after rollback (within thresholds for {guard.RECOVERY_SAMPLES} samples)")
    else:
        why = recovery.get("note") or f"within {guard.RECOVERY_TIMEOUT_S}s"
        print(f"\nRecovery:   did not recover ({why})")
    print(f"\nHypothesis: {plan['hypothesis']}\nVerdict:    {verdict['verdict'].upper()} - {verdict['explanation']}\n{line}")


def _rounded(x):
    """Round floats for the LLM payload (error rates are in %, latencies in ms)."""
    if isinstance(x, float):
        return round(x, 2)
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    return [_rounded(v) for v in x] if isinstance(x, list) else x


def run(plan, source, validation):
    """Inject, guard, always roll back, wait for recovery, summarise and save."""
    baseline = load_baseline()
    if baseline is None:
        print("No baseline.json (run metrics.py first). Not injecting.")
        return 1
    services = guard.watched_services(plan, baseline, validation["blast_radius"])
    if not services:
        print("None of the affected services have baseline thresholds; cannot guard. Not injecting.")
        return 1
    thresholds = {s: FORCE_THRESHOLDS for s in services} if FORCE_ABORT_TEST else None
    try:  # never inject if the guard cannot see
        guard.live_metrics(services)
    except Exception as e:
        print(f"Telemetry (Jaeger) is not reachable, so the guard would be blind: {e}. Not injecting.")
        return 1
    if baseline["flags_enabled"]:
        print(f"Note: baseline was captured with flags on ({', '.join(baseline['flags_enabled'])}).")
    if FORCE_ABORT_TEST:
        print("*** FORCE_ABORT_TEST: tiny thresholds, expect an automatic abort ***")
    print(f"\nWatching: {', '.join(services)}")

    started = datetime.now(timezone.utc)
    g = {"aborted": True, "reason": "not started", "elapsed_s": 0, "timeline": [],
         "peaks": {s: {m: None for m in guard.METRICS} for s in services}}
    try:
        active = inject.inject(plan)
        print(f"FAULT ACTIVE: {plan['fault']} on {active['container_name']} (rollback: {active['rollback']})", flush=True)
        g = guard.watch(plan, baseline, services, thresholds)
    except KeyboardInterrupt:
        g["reason"] = "interrupted by user (Ctrl+C) during injection"
    except Exception as e:
        g["reason"] = f"error: {type(e).__name__}: {e}"
    finally:
        rolled_back = _rollback()
    print(f"\n{'ABORTED' if g['aborted'] else 'Fault window finished'}: {g['reason']}")

    recovery = {"recovered": False, "seconds": None, "note": "rollback failed"}
    if rolled_back:
        print(f"Waiting up to {guard.RECOVERY_TIMEOUT_S}s for recovery...")
        recovery = guard.wait_for_recovery(plan, baseline, services)
    impact = guard.impact_table(baseline, g["peaks"])
    containers = guard.container_table(baseline, g.get("container_peaks", {}))
    measured = {"aborted": g["aborted"], "reason": g["reason"], "elapsed_s": g["elapsed_s"],
                "impact_baseline_to_peak": _rounded(impact), "recovery": recovery,
                "forced_abort_test": FORCE_ABORT_TEST, "baseline_flags_enabled": baseline["flags_enabled"]}
    verdict = judge_hypothesis(plan, measured)
    print_results(plan, g, rolled_back, recovery, impact, containers, verdict)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps({
        "plan": {k: v for k, v in plan.items() if k != "ranking"}, "plan_source": source,
        "ranking": plan.get("ranking"), "validation": validation,
        "started_at": started.isoformat(timespec="seconds"),
        "baseline": {"captured_at": baseline["captured_at"], "flags_enabled": baseline["flags_enabled"]},
        "forced_abort_test": FORCE_ABORT_TEST, "guard": g, "rollback_verified": rolled_back,
        "recovery": recovery, "impact": impact, "container_resources": containers, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    return 0 if rolled_back else 2


def main():
    if not _recover_crashed_run():
        return 2
    plan, source, note = propose_experiment()
    result = validate(plan)
    print_summary(plan, source, note, result)
    if not result["valid"]:
        print("Plan rejected by policy; nothing to approve.")
        return 1
    try:
        answer = input("Approve this experiment? (y/n) ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Rejected")
        return 0
    APPROVED_JSON.write_text(json.dumps({
        "plan": plan, "source": source, "blast_radius": result["blast_radius"],
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"Approved - starting experiment (plan saved to {APPROVED_JSON})")
    return run(plan, source, result)


if __name__ == "__main__":
    sys.exit(main())
