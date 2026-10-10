"""Small dependency graph, baseline and history shared by the ranker and agent tests.

    web -> checkout -> pay            (sync)
           checkout -> queue <- worker (async only)
    web, checkout -> flagd             (protected)
    web -> ghost                       (no running container)
"""
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYNC = {"mode": "sync", "type": "sync-grpc"}
ASYNC = {"mode": "async", "type": "async-messaging"}


def graph():
    g = nx.DiGraph()
    g.add_edge("web", "checkout", **SYNC)
    g.add_edge("checkout", "pay", **SYNC)
    g.add_edge("checkout", "queue", **ASYNC)
    g.add_edge("worker", "queue", **ASYNC)
    g.add_edge("web", "flagd", **SYNC)
    g.add_edge("checkout", "flagd", **SYNC)
    g.add_edge("web", "ghost", **SYNC)
    return g


def _stat(mean, std=0.0):
    return {"mean": mean, "std": std, "max": mean + std, "n": 20}


def baseline():
    svc = lambda rate, p95, std=1.0: {"request_rate": _stat(rate), "error_rate": _stat(0.0), "p95_ms": _stat(p95, std)}
    running = {"running": _stat(1)}
    return {"services": {"web": svc(2.0, 30), "checkout": svc(0.5, 60), "pay": svc(0.5, 5),
                         "worker": svc(0.5, 20), "flagd": svc(3.0, 2)},
            "containers": {n: running for n in ("web", "checkout", "pay", "queue", "worker", "flagd")},
            "thresholds": {"services": {}, "containers": {}}, "requests": {}}


def flows():
    steps = [("web", "checkout"), ("checkout", "pay"), ("checkout", "queue"), ("worker", "queue")]
    return {"buy": {"steps": [{"from": a, "to": b} for a, b in steps]}}


def run(target, fault, magnitude, verdict="held", guard_aborted=False, recovery_s=10.0):
    return {"file": "x.json", "target": target, "fault": fault, "magnitude": magnitude, "verdict": verdict,
            "guard_aborted": guard_aborted, "reason": "", "recovery_s": recovery_s}
