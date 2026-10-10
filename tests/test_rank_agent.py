import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixtures import baseline, flows, graph

import rank_agent
from policy import validate


class SearchSourceSandboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "demo"
        (self.root / "src" / "checkout").mkdir(parents=True)
        (self.root / "src" / "checkout" / "main.go").write_text(
            "func charge(ctx context.Context) {\n\tctx, cancel := context.WithTimeout(ctx, time.Second)\n}\n")
        self.outside = base / "outside"
        self.outside.mkdir()
        (self.outside / "secret.go").write_text("WithTimeout secret\n")
        env = mock.patch.dict(os.environ, {rank_agent.SRC_ENV_VAR: str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_finds_matches_inside_root(self):
        out = rank_agent.search_source("checkout", "WithTimeout")
        self.assertEqual([m["at"] for m in out["matches"]], ["src/checkout/main.go:2"])

    def test_refuses_paths_outside_root(self):
        for bad in ("../outside", "..", "../../outside", str(self.outside), "checkout/../../outside", "/etc", "C:\\Windows"):
            with self.assertRaises(rank_agent.ToolError, msg=bad):
                rank_agent.search_source(bad, "WithTimeout")
        # the LLM-facing dispatcher turns the refusal into an error result instead of raising
        tools = rank_agent.Toolbox(graph(), baseline(), [], [])
        self.assertIn("refused", tools.call("search_source", {"service": "../outside", "pattern": "x"})["error"])

    def test_does_not_follow_symlinks_out_of_root(self):
        try:
            os.symlink(self.outside, self.root / "src" / "checkout" / "link", target_is_directory=True)
            os.symlink(self.outside / "secret.go", self.root / "src" / "checkout" / "secret.go")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlinks here (Windows without developer mode)")
        out = rank_agent.search_source("checkout", "secret")
        self.assertEqual(out["matches"], [])

    def test_refuses_without_root(self):
        with mock.patch.dict(os.environ, {rank_agent.SRC_ENV_VAR: ""}), mock.patch.object(rank_agent, "load_dotenv"):
            with self.assertRaises(rank_agent.ToolError):
                rank_agent.search_source("checkout", "x")


def reply(*parts):
    return {"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP"}


class RankAgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "agent_log.jsonl"
        self.g = graph()

    def agent(self, replies):
        it = iter(replies)
        post = lambda api_key, body, models: (next(it), "fake-model")
        return rank_agent.run_agent(api_key="test-key", post=post, graph=self.g, baseline=baseline(),
                                    flows=flows(), history=[], log_path=self.log)

    def log_types(self):
        return [json.loads(line)["type"] for line in self.log.read_text().splitlines()]

    def test_garbage_output_falls_back_to_ranker_top(self):
        out = self.agent([reply({"text": "I think payment is risky!!! <html>"})])
        self.assertEqual(out["source"], "ranker_fallback")
        self.assertIn("LLM investigation failed", out["note"])
        self.assertEqual(out["chosen"], out["top_k"][0]["candidate_id"])
        self.assertTrue(validate(out["plan"], self.g)["valid"])
        self.assertEqual(self.log_types()[-1], "decision")

    def test_choice_outside_top_k_falls_back(self):
        bad = {"ranking": [], "chosen": "flagd:container_stop:0", "hypothesis": "h", "predictions": []}
        out = self.agent([reply({"text": json.dumps(bad)})])
        self.assertEqual(out["source"], "ranker_fallback")
        self.assertIn("not one of the top", out["note"])

    def test_llm_exception_falls_back(self):
        def post(api_key, body, models):
            raise RuntimeError("all models failed")
        out = rank_agent.run_agent(api_key="k", post=post, graph=self.g, baseline=baseline(), flows=flows(),
                                   history=[], log_path=self.log)
        self.assertEqual(out["source"], "ranker_fallback")

    def test_tool_call_then_valid_answer(self):
        top = [c["id"] for c in rank_agent.top_k(rank_agent.ranker.rank(self.g, baseline(), flows(), []), 3)]
        answer = {"ranking": [{"candidate_id": cid, "adjusted_risk": 0.5, "adjusted_priority": 0.4,
                               "evidence": [{"tool": "get_edges", "finding": "one sync caller"}], "reason": "r"}
                              for cid in top],
                  "chosen": top[1], "hypothesis": "callers slow down and recover",
                  "predictions": [{"svc": "checkout", "metric": "p95_ms", "direction": "up", "min_factor": 1.5}]}
        out = self.agent([reply({"functionCall": {"name": "get_edges", "args": {"service": "pay"}}}),
                          reply({"text": json.dumps(answer)})])
        self.assertEqual(out["source"], "rank_agent")
        self.assertEqual(out["chosen"], top[1])
        self.assertEqual(out["tool_calls"], 1)
        self.assertEqual(out["plan"]["hypothesis"], "callers slow down and recover")
        self.assertEqual(self.log_types(), ["thought", "tool_call", "tool_result", "decision"])

    def test_tool_budget_is_enforced(self):
        loop = [reply({"functionCall": {"name": "get_history", "args": {"service": "pay"}}})] * 20
        out = self.agent(loop)
        self.assertEqual(out["source"], "ranker_fallback")
        self.assertEqual(self.log_types().count("tool_call"), rank_agent.MAX_TOOL_CALLS)


if __name__ == "__main__":
    unittest.main()
