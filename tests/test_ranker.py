import copy
import unittest

from fixtures import baseline, flows, graph, run

import ranker
from policy import validate


def best(ranking, target):
    return min((c for c in ranking if c["target"] == target), key=lambda c: c["rank"])


def find(ranking, cid):
    return next(c for c in ranking if c["id"] == cid)


class RankerTest(unittest.TestCase):
    def setUp(self):
        self.g, self.base, self.flows = graph(), baseline(), flows()

    def rank(self, history=(), weights=ranker.WEIGHTS):
        return ranker.rank(self.g, self.base, self.flows, list(history), weights=weights)

    def test_protected_and_stopped_services_are_excluded(self):
        ranking = self.rank()
        targets = {c["target"] for c in ranking}
        self.assertNotIn("flagd", targets)   # protected
        self.assertNotIn("ghost", targets)   # no running container in the baseline
        self.assertEqual(targets, {"web", "checkout", "pay", "queue", "worker"})
        for c in ranking:  # every candidate passes policy and has a why-line
            plan = {**{k: c[k] for k in ("target", "fault", "magnitude", "duration_s")},
                    "hypothesis": "h", "reasoning": "r"}
            self.assertTrue(validate(plan, self.g)["valid"], c["id"])
            self.assertTrue(c["why"])

    def test_untested_sync_checkout_service_outranks_tested_async_only(self):
        ranking = self.rank([run("queue", "container_stop", 0)])
        pay, queue = best(ranking, "pay"), best(ranking, "queue")
        self.assertTrue(pay["features"]["checkout_path"]["raw"])
        self.assertEqual(queue["features"]["sync_callers"]["raw"], 0)
        self.assertLess(pay["rank"], queue["rank"])
        self.assertGreater(pay["risk"], queue["risk"])

    def test_exact_repeat_gets_lower_informativeness(self):
        fresh = find(self.rank(), "pay:network_delay:500")
        ranking = self.rank([run("pay", "network_delay", 500)])
        repeat = find(ranking, "pay:network_delay:500")
        self.assertLess(repeat["info"], fresh["info"])
        self.assertLess(repeat["info"], find(ranking, "pay:network_delay:200")["info"])
        self.assertEqual(repeat["features"]["novelty"]["raw"], "exact_repeat")
        self.assertIn("exact repeat (x1)", repeat["why"])
        # a smaller step that held makes the next step up an escalation
        self.assertEqual(find(ranking, "pay:network_delay:1000")["features"]["escalation"]["score"], 1.0)

    def test_weights_change_the_order(self):
        # pay is high-risk but fully explored; worker is low-risk and never tested
        history = [run("pay", "container_stop", 0)] + [run("pay", "network_delay", m) for m in (200, 500, 1000)]
        risk_only, info_only = copy.deepcopy(ranker.WEIGHTS), copy.deepcopy(ranker.WEIGHTS)
        risk_only["priority"] = {"risk": 1.0, "info": 0.0, "safety": 0.0, "prior_abort": 0.0}
        info_only["priority"] = {"risk": 0.0, "info": 1.0, "safety": 0.0, "prior_abort": 0.0}
        by_risk, by_info = self.rank(history, risk_only), self.rank(history, info_only)
        self.assertLess(best(by_risk, "pay")["rank"], best(by_risk, "worker")["rank"])
        self.assertLess(best(by_info, "worker")["rank"], best(by_info, "pay")["rank"])

    def test_guard_abort_penalises_same_or_bigger_magnitude(self):
        ranking = self.rank([run("pay", "network_delay", 500, guard_aborted=True)])
        self.assertEqual(find(ranking, "pay:network_delay:1000")["features"]["prior_abort"]["score"], 1.0)
        self.assertEqual(find(ranking, "pay:network_delay:200")["features"]["prior_abort"]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
