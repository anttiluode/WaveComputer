"""Is the fast machine the same machine as QualiaClaude's node-by-node engines?

Three checks, from the parts to the whole:

  1. nodes   the vectorized FadeClocks and Branches against the vendored FadingHistorySpikeNode and
             DampingReturnNode, fed identical random inputs: spikes and weights must be bit-identical.
  2. physics the kernel field against the vendored 96x96 WaveFieldNode, fed identical random writes:
             equal to float32 rounding (the node runs in float32, the kernel sums in float64).
  3. bodies  the same body trained for 240 episodes through both engines (content = relay/, residue =
             innovation/), then read with eyes open and closed (and regen): the C readouts must agree
             within 2% of their own size. Rounding (check 2) can move a clock that sits within ~1e-7 of its
             threshold by one tick, so whole bodies are compared on what is measured, not spike by spike.

    python tests/test_equivalence.py          checks 1 and 2 (seconds)
    python tests/test_equivalence.py --body   also check 3 (about 8 minutes: the reference engine is slow)
"""
from __future__ import annotations
import pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = ROOT / "reference" / "qualiaclaude"
sys.path[:0] = [str(ROOT), str(REF), str(REF / "relay"), str(REF / "innovation")]
import engine as E
from wavecomputer import machine as M, physics as P, nodes as Nd


def check_nodes(seed=11):
    rng = np.random.default_rng(seed)
    n, T = 24, 400
    decay = rng.uniform(0.85, 0.98, n); arm = rng.uniform(0.02, 0.1, n); rel = arm * rng.uniform(0.1, 0.5, n)
    ref = [E.make("FadingHistorySpikeNode", {"decay": d, "arm_threshold": a, "release_threshold": r,
                                              "pulse_amplitude": 0.24, "refractory": 7, "enabled": 1})
           for d, a, r in zip(decay, arm, rel)]
    fast = Nd.FadeClocks(decay, arm, rel, 0.24, 7)
    W0 = rng.normal(0, 0.3, (n, 8))
    rule = dict(M.RULE, learning_rate=0.05)
    gref = [E.make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=1)) for w in W0]
    gfast = Nd.Branches(W0, rule["learning_rate"], rule["listen_from"], rule["listen_to"], rule["self_response"],
                        rule["max_weight"], rule["sign"], learn=True)
    x = np.zeros((T, n))                                  # bursts, then silence: clocks arm, fade, fire
    for k in range(n):
        t = int(rng.integers(0, 20))
        while t < T:
            x[t:t + 3, k] = rng.uniform(0.15, 0.4); t += int(rng.integers(15, 45))
    site = rng.normal(0, 0.1, (T, n, 8))
    same = True; prev = np.zeros(n)
    for t in range(T):
        for k, u in enumerate(ref):
            u.pre_step(); u.set_input("signal_in", x[t, k]); u.step()
        fast.step(x[t])
        pr = np.array([u.pulse for u in ref])
        same &= np.array_equal(pr, fast.pulse)
        for k, g in enumerate(gref):
            g.pre_step(); g.set_input("pre_spike", prev[k])
            for s in range(8):
                g.set_input("site_%d" % s, site[t, k, s])
            g.step()
        gfast.step(prev, site[t])
        same &= np.array_equal(np.array([g.weights for g in gref]), gfast.W)
        prev = pr
    fired = int(sum(u.fire_count for u in ref))
    print("1. nodes    %d clocks x %d ticks (%d pulses), branches learning: %s"
          % (n, T, fired, "bit-identical" if same else "DIFFERENT"))
    return bool(same)


def check_physics(seed=12, T=240):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.05, (T, 8)) * (rng.random((T, 8)) < 0.15)
    f = P.vendored_field(); fast = P.Field(T); a = np.zeros((T, 8)); b = np.zeros((T, 8))
    for t in range(T):
        f.pre_step()
        for s in range(8):
            if w[t, s]:
                f.set_input("write_%d" % s, w[t, s])
        f.step(); a[t] = f.read_values; b[t] = fast.step(w[t])
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    print("2. physics  %d ticks of random writes at all 8 sites: max difference %.1e of the field's size" % (T, rel))
    return rel < 1e-5


def check_bodies():
    import engine_relay as L, engine_regen as G, engine_innov as I, run_relay as R
    U = R.U(); MU = M.Units(); ok = True
    for kind, seed, rel in (("content", 180001, (1, 1)), ("residue", 190001, (1, -1))):
        t0 = time.time()
        gap, amp, eps = R.schedule(seed)
        Wr = [[0.0] * 8 for _ in U]; Wm = np.zeros((MU.n, 8))
        for sa, k1, k2, onset, g, g2 in eps:
            sb = rel[0] * sa * (1 if k1 else -1); sc = rel[1] * sb * (1 if k2 else -1)
            ev = R.events(sa, sb, sc, onset, g, g2, amp)
            if kind == "content":
                Wr = L.run(U, Wr, ev, R.STEPS, learn=True, rule=R.RULE, mode="residual")["weights"]
            else:
                Wr = I.run(U, Wr, ev, R.STEPS, learn=True, rule=R.RULE, b_arm=0.8)["weights"]
            Wm = M.run(MU, Wm, ev, R.STEPS, learn=True, bkind=kind)["weights"]
        g2 = gap + 20; c0 = R.PROBE_ONSET + gap + g2
        reads = [("eyes open, A then B", dict(eyes="open"), (1, rel[0], 0)),
                 ("eyes closed, A alone", dict(eyes="closed"), (1, 0, 0))]
        for name, kw, (sa, sb, sc) in reads:
            ev = R.events(sa, sb, sc, R.PROBE_ONSET, gap, g2)
            ra = (L.run(U, Wr, ev, R.STEPS, gain=1.9, **kw) if kind == "content" else I.run(U, Wr, ev, R.STEPS, **kw))
            rb = M.run(MU, Wm, ev, R.STEPS, bkind=kind, gain=1.9, **kw)
            ca, cb = R.proj(np.asarray(ra["prediction"]), "C", c0, 40), R.proj(rb["prediction"], "C", c0, 40)
            d = abs(ca - cb) / max(abs(ca), 1e-9)
            ok &= d < 0.02
            print("3. body     %-8s seed %d, %-22s C = %+.4f (nodes) vs %+.4f (fast): %.2f%%" % (kind, seed, name, ca, cb, 100 * d))
        if kind == "content":
            ev = R.events(1, 0, 0, R.PROBE_ONSET, gap, g2)
            ra = G.run(U, Wr, ev, R.STEPS, gain=1.9); rb = M.run(MU, Wm, ev, R.STEPS, bkind="regen", eyes="closed", gain=1.9)
            ca, cb = R.proj(np.asarray(ra["prediction"]), "C", c0, 40), R.proj(rb["prediction"], "C", c0, 40)
            d = abs(ca - cb) / max(abs(ca), 1e-9); ok &= d < 0.02
            print("3. body     %-8s seed %d, %-22s C = %+.4f (nodes) vs %+.4f (fast): %.2f%%" % ("regen", seed, "eyes closed, A alone", ca, cb, 100 * d))
        print("           (%.0f s)" % (time.time() - t0))
    return ok


if __name__ == "__main__":
    ok = check_nodes() and check_physics()
    if "--body" in sys.argv:
        ok = check_bodies() and ok
    print("EQUIVALENT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
