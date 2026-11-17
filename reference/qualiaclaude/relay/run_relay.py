"""Relay experiment (see relay/PROTOCOL_RELAY.md).

  python relay/run_relay.py --check     kernel check (own contribution == field) and unit calibration
  python relay/run_relay.py --dev       development bodies (seeds 981-983)
  python relay/run_relay.py             the pre-registered batch: 16 bodies x 2 rules
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import engine as E
import engine_relay as L

RULE = {"learning_rate": 0.03, "listen_from": 3, "listen_to": 5, "self_response": 0.71, "max_weight": 2.0, "sign": 1.0}
Q = 0.85
EPISODES, STEPS, PROBE_ONSET, GAP2_EXTRA = 240, 240, 23, 20
RELATIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
SEEDS = list(range(180001, 180017))
DEV_SEEDS = [981, 982, 983]
RULES = ("residual", "similarity")
FIRE = {"AB": np.linspace(24, 60, 12), "AC": np.linspace(66, 122, 12), "B": np.linspace(40, 78, 12)}
ARM, REL = 0.80, 0.20
PERM = [(5 * k) % 12 for k in range(12)]              # graded thresholds spread across clocks, not by clock time
THETA = [0.25 + 0.75 * (PERM[k] + 0.5) / 12 for k in range(12)]   # over (0.25, 1): below ~0.2 wrong-sign units armed on the imagined B's opposite lobe (dev)


def fire_time(decay, pop):
    onset, st = 20, L.HEARS[pop]
    w = E.passive([(L.SITES[st], onset, 1.0)], 300)[:, L.READ[st]]
    p = E.peak_trace(decay, np.maximum(w, 0))
    u = L.Unit(decay, (ARM * p, REL * p), pop, 1)
    sp = L.run([u], [[0.0] * 8], [(L.SITES[st], onset, 1.0)], 300)["spikes"][0]
    return (sp[0] - onset) if sp else None, (ARM * p, REL * p)


def calibrate(target, pop):
    lo, hi = 0.70, 0.995
    for _ in range(26):
        mid = (lo + hi) / 2
        t, _ = fire_time(mid, pop)
        if t is None or t > target:
            hi = mid
        else:
            lo = mid
    t, th = fire_time(lo, pop)
    return lo, th, t


_U = None
CAL_FILE = HERE / "units_relay.json"
def U():
    global _U
    if _U is None:
        if CAL_FILE.exists():
            cal = {p: [(d, tuple(th), t) for d, th, t in rows] for p, rows in json.loads(CAL_FILE.read_text()).items()}
        else:
            cal = {p: [calibrate(t, p) for t in FIRE[p]] for p in ("AB", "AC", "B")}
            CAL_FILE.write_text(json.dumps({p: [[d, list(th), t] for d, th, t in rows] for p, rows in cal.items()}, indent=1))
        _U = [L.Unit(d, th, pop, half, THETA[i]) for pop in ("AB", "AC", "B") for half in (1, -1)
              for i, (d, th, _) in enumerate(cal[pop])]
        U.cal = cal
    return _U


def schedule(seed):
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(.9, 1.12))
    sa = np.array([1] * (EPISODES // 2) + [-1] * (EPISODES // 2)); rng.shuffle(sa)
    eps = []
    for a in sa:
        k1, k2 = rng.random() < Q, rng.random() < Q
        onset = 20 + int(rng.integers(-5, 6)); g = gap + int(rng.integers(-3, 4))
        eps.append((int(a), bool(k1), bool(k2), onset, g, gap + GAP2_EXTRA))
    return gap, amp, eps


def events(sa, sb, sc, onset, g, g2, amp=1.0):
    ev = [(L.SITES["A"], onset, amp * sa)]
    if sb:
        ev.append((L.SITES["B"], onset + g, amp * sb))
    if sc:
        ev.append((L.SITES["C"], onset + g + g2, amp * sc))
    return ev


def train(seed, rel, mode):
    gap, amp, eps = schedule(seed)
    W = [[0.0] * 8 for _ in U()]
    for sa, k1, k2, onset, g, g2 in eps:
        sb = rel[0] * sa * (1 if k1 else -1); sc = rel[1] * sb * (1 if k2 else -1)
        W = L.run(U(), W, events(sa, sb, sc, onset, g, g2, amp), STEPS, learn=True, rule=RULE, mode=mode)["weights"]
    return gap, W


def proj(P, site, t0, n):
    tpl = E.passive([(L.SITES[site], t0, 1.0)], STEPS)[t0:t0 + n].ravel()
    return float(P[t0:t0 + n].ravel() @ tpl / (tpl @ tpl))


def fired(r):
    out = {}
    for pop in ("AB", "AC", "B"):
        for h in (1, -1):
            out["%s%+d" % (pop, h)] = int(sum(1 for k, u in enumerate(U()) if u.pop == pop and u.half == h and r["spikes"][k]))
    return out


def read(W, gap, rel):
    r1, r2 = rel
    g2 = gap + GAP2_EXTRA
    b0, c0 = PROBE_ONSET + gap, PROBE_ONSET + gap + g2
    out = {"open": {}, "closed_normalized": {}, "closed_graded": {}}
    # eyes open, A with and without the real B
    bc = {}
    for sa in (1, -1):
        for sb in (1, -1, 0):
            r = L.run(U(), W, events(sa, sb, 0, PROBE_ONSET, gap, g2), STEPS)
            bc[(sa, sb)] = proj(r["prediction"], "C", c0, 40)
            out["open"]["fired%+d%+d" % (sa, sb)] = fired(r)
    out["open"]["A_given_B"] = float(np.mean([(bc[(1, sb)] - bc[(-1, sb)]) / 2 for sb in (1, -1)]) * r1 * r2)
    out["open"]["B_influence"] = float(np.mean([(bc[(sa, 1)] - bc[(sa, -1)]) / 2 for sa in (1, -1)]) * r2)
    out["open"]["C_from_A_alone"] = float((bc[(1, 0)] - bc[(-1, 0)]) / 2 * r1 * r2)
    # step one: the imagined B (B-units silent), and the body's own gain G
    bb, pk = {}, {}
    for sa in (1, -1):
        r = L.run(U(), W, events(sa, 0, 0, PROBE_ONSET, gap, g2), STEPS, cut=("B",))
        bb[sa] = proj(r["prediction"], "B", b0, 30)
        pk[sa] = float(np.max(np.abs(r["prediction"][b0 - 5:b0 + 30, L.READ["B"]])))
    real_peak = float(np.max(np.abs(E.passive([(L.SITES["B"], b0, 1.0)], STEPS)[:, L.READ["B"]])))
    gain = real_peak / max(np.mean(list(pk.values())), 1e-9)
    out["step_one"] = float((bb[1] - bb[-1]) / 2 * r1)
    out["imagined_strength"] = 1.0 / gain
    out["gain"] = gain
    # eyes closed: B-units hear the imagined B
    seen = L.run(U(), W, events(1, r1, 0, PROBE_ONSET, gap, g2), STEPS)["spikes"]
    bk = [k for k, u in enumerate(U()) if u.pop == "B" and u.half == r1]
    for name in ("normalized", "graded"):
        cc = {}
        for sa in (1, -1):
            r = L.run(U(), W, events(sa, 0, 0, PROBE_ONSET, gap, g2), STEPS, eyes="closed", imagine=name, gain=gain)
            cc[sa] = proj(r["prediction"], "C", c0, 40)
            out["closed_" + name]["fired%+d" % sa] = fired(r)
            if sa == 1:
                lag = [r["spikes"][k][0] - seen[k][0] for k in bk if r["spikes"][k] and seen[k]]
                out["closed_" + name]["B_clock_lag_vs_seen"] = lag
        out["closed_" + name]["C_from_A"] = float((cc[1] - cc[-1]) / 2 * r1 * r2)
    return out


def job(arg):
    seed, rel, mode = arg
    gap, W = train(seed, rel, mode)
    return {"seed": seed, "rule": mode, "rho1": rel[0], "rho2": rel[1], "gap": gap, "weights": W, **read(W, gap, rel)}


def check():
    W = [[0.0] * 8 for _ in U()]
    k = next(i for i, u in enumerate(U()) if u.pop == "AB" and u.half == 1)
    W[k][3], W[k][4], W[k][5] = 0.5, -0.3, 0.2
    ev = events(1, 0, 0, 20, 34, 54)
    r = L.run(U()[k:k + 1], W[k:k + 1], ev, STEPS)
    t = r["spikes"][0][0]
    own = [L.E.PING_AMP * sum(W[k][sw] * L.kernel()[sw, L_ - 3] for sw in (3, 4, 5)) for L_ in range(3, 3 + L.K_LEN)]
    field = [r["prediction"][t + L_ - 1] for L_ in range(3, 3 + L.K_LEN)]
    d = max(float(np.abs(a - b).max()) for a, b in zip(own, field))
    print("own vs field max difference:", d); assert d < 1e-6
    for pop, rows in U.cal.items():
        print(pop, "fire", [x[2] for x in rows])
    print("theta by clock", [round(x, 3) for x in THETA])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    t0 = time.time(); U()
    if a.dev:
        jobs = [(s, (1, 1), m) for s in DEV_SEEDS for m in RULES]
        out, label = HERE / "dev_results_relay.json", "relay: development bodies (not evidence)"
    else:
        jobs = [(s, RELATIONS[i % 4], m) for m in RULES for i, s in enumerate(SEEDS)]
        out, label = HERE / "results_relay.json", "relay: pre-registered batch"
    receipt = {"label": label, "rule": RULE, "q": Q, "episodes": EPISODES, "theta": THETA, "results": [],
               "runner_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
               "engine_sha256": hashlib.sha256((HERE / "engine_relay.py").read_bytes()).hexdigest()}
    if not a.dev:
        receipt["protocol_sha256"] = hashlib.sha256((HERE / "PROTOCOL_RELAY.md").read_bytes()).hexdigest()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for n, rec in enumerate(pool.map(job, jobs), 1):
            receipt["results"].append(rec); receipt["elapsed_seconds"] = time.time() - t0
            out.write_text(json.dumps(receipt, indent=1))
            print("%d/%d %.0fs" % (n, len(jobs), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
