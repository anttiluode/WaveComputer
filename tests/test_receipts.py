"""Retrain every published QualiaClaude body of relay/ and innovation/ in the fast machine, from the same
schedules, and compare with the final weights stored in those receipts (reference/receipts/).

Sensor waves are taken bit-exact from the float32 node (physics.EXACT_SENSORS). The prediction field is
the float64 kernel sum, about 1e-7 from the node: weights learned from it differ at that level, and a unit
that listens to the prediction field (innovation's B units) can still meet a threshold tie.

    python tests/test_receipts.py        (about 5 minutes on two cores)
"""
from __future__ import annotations
import json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "qualiaclaude"
sys.path[:0] = [str(ROOT), str(REF), str(REF / "relay")]
import run_relay as R
from wavecomputer import machine as M, physics as P


def schedule_innov(seed, q1):                    # innovation/run_innov.py's schedule, verbatim
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(.9, 1.12))
    sa = np.array([1] * (R.EPISODES // 2) + [-1] * (R.EPISODES // 2)); rng.shuffle(sa)
    eps = []
    for a in sa:
        k1, k2 = rng.random() < q1, rng.random() < 0.85
        onset = 20 + int(rng.integers(-5, 6)); g = gap + int(rng.integers(-3, 4))
        eps.append((int(a), bool(k1), bool(k2), onset, g, gap + R.GAP2_EXTRA))
    return gap, amp, eps


def job(rec):
    P.EXACT_SENSORS = True             # sensor waves bit-identical to the float32 node (see physics.py)
    U = M.Units()
    if rec["src"] == "relay":
        gap, amp, eps = R.schedule(rec["seed"]); kind = "content"
    else:
        gap, amp, eps = schedule_innov(rec["seed"], rec["q1"]); kind = "residue"
    W = np.zeros((U.n, 8)); rel = (rec["rho1"], rec["rho2"])
    for sa, k1, k2, onset, g, g2 in eps:
        sb = rel[0] * sa * (1 if k1 else -1); sc = rel[1] * sb * (1 if k2 else -1)
        W = M.run(U, W, R.events(sa, sb, sc, onset, g, g2, amp), R.STEPS, learn=True, bkind=kind)["weights"]
    pub = np.asarray(rec["weights"], float)
    return rec["src"], rec["seed"], float(np.max(np.abs(W - pub))), float(np.max(np.abs(pub)))


if __name__ == "__main__":
    recs = []
    for r in json.loads((ROOT / "reference/receipts/results_relay.json").read_text())["results"]:
        if r["rule"] == "residual":
            recs.append(dict(r, src="relay"))
    for r in json.loads((ROOT / "reference/receipts/results_innov.json").read_text())["results"]:
        if r["body"] == "innovation":
            recs.append(dict(r, src="innovation"))
    with ProcessPoolExecutor(2) as pool:
        res = list(pool.map(job, recs))
    for src in ("relay", "innovation"):
        d = [x for x in res if x[0] == src]
        dif = np.array([x[2] for x in d]); size = np.array([x[3] for x in d])
        print("%-10s %2d published bodies: largest weight difference %.1e (weights up to %.2f); bodies within 1e-6: %d/%d"
              % (src, len(d), dif.max(), size.max(), int((dif < 1e-6).sum()), len(d)))
