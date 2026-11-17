"""One body: every capability of the Qualia day, asked of the same trained body.

Worlds (A, B, C are +-1 events; rel = (rho1, rho2) is the relation type, cycled over the 4 types):
  true B = rho1 * A, flipped with probability 1 - q1
  true C = rho2 * true B, flipped with probability 1 - q2
  seen B = true B, flipped with probability 1 - r, its size times U(1 - z, 1 + z)   (clean: r = 1, z = 0)
The body only ever sees the seen B; C is caused by the true B.

Exact answers (Bayes), with m_A = 2 q1 - 1, m_b = 2 r - 1, k = 2 q2 - 1:
  eyes open, seen B agrees with A      E[C] along B  = k (m_A + m_b) / (1 + m_A m_b)
  eyes open, seen B contradicts A      E[C] along B  = k (m_b - m_A) / (1 - m_A m_b)
  eyes closed (no B), A alone          E[C] along A  = k m_A
With clean B (m_b = 1) the first two are k and k: a seen B settles C whatever A said.

Two trainings per world and seed (same schedule):
  "content" body: B units hear the seen B. Read with three B-unit kinds eyes closed: content, regen,
                  twoform (they differ only in imagination; with eyes open they are the same unit).
  "residue" body: B units hear seen B minus the body's own prediction (innovation/).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wavecomputer import machine as M
from wavecomputer.physics import passive

EPISODES, STEPS, PROBE, GAP2_EXTRA = 240, 240, 23, 20
RELATIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
WORLDS = {
    "clean85": dict(q1=0.85, q2=0.85, r=1.0, z=0.0),
    "clean70": dict(q1=0.70, q2=0.85, r=1.0, z=0.0),
    "noisy85": dict(q1=0.85, q2=0.85, r=0.85, z=0.2),
}
_U = None
def U():
    global _U
    if _U is None:
        _U = M.Units()
    return _U


def bayes(w):
    mA, mb, k = 2 * w["q1"] - 1, 2 * w["r"] - 1, 2 * w["q2"] - 1
    return {"agree": k * (mA + mb) / (1 + mA * mb), "contradict": k * (mb - mA) / (1 - mA * mb), "closed": k * mA,
            "k": k, "mA": mA}


def schedule(seed, w):
    rng = np.random.default_rng(seed)
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(.9, 1.12))
    sa = np.array([1] * (EPISODES // 2) + [-1] * (EPISODES // 2)); rng.shuffle(sa)
    eps = []
    for a in sa:
        k1, k2, kr = rng.random() < w["q1"], rng.random() < w["q2"], rng.random() < w["r"]
        size = float(rng.uniform(1 - w["z"], 1 + w["z"])) if w["z"] else 1.0
        onset = 20 + int(rng.integers(-5, 6)); g = gap + int(rng.integers(-3, 4))
        eps.append((int(a), bool(k1), bool(k2), bool(kr), size, onset, g, gap + GAP2_EXTRA))
    return gap, amp, eps


def train(seed, w, rel, kind):
    gap, amp, eps = schedule(seed, w)
    W = np.zeros((U().n, 8))
    bk = "residue" if kind == "residue" else "content"
    for sa, k1, k2, kr, size, onset, g, g2 in eps:
        tb = rel[0] * sa * (1 if k1 else -1)                 # true B
        sc = rel[1] * tb * (1 if k2 else -1)                 # C, caused by the true B
        sb = tb * (1 if kr else -1) * size                   # the B the body sees
        W = M.run(U(), W, M.events(sa, sb, sc, onset, g, g2, amp), STEPS, learn=True, bkind=bk)["weights"]
    return gap, W


def proj(P, site, t0, n):
    tpl = passive([(M.SITES[site], t0, 1.0)], STEPS)[t0:t0 + n].ravel()
    return float(P[t0:t0 + n].ravel() @ tpl / (tpl @ tpl))


def spikes_in(r, pop, half):
    return int(sum(1 for k in U().index(pop, half) if r["spikes"][k]))


def read(W, gap, rel, w, kind):
    """Every readout, sign-aligned so that + is the direction the exact answer points."""
    r1, r2 = rel
    g2 = gap + GAP2_EXTRA; b0 = PROBE + gap; c0 = b0 + g2
    bk = "residue" if kind == "residue" else "content"
    ev = lambda sa, sb: M.events(sa, sb, 0, PROBE, gap, g2)
    out = {}
    # eyes open: A with a seen B that agrees or contradicts, and A with B absent
    C, res, bsp = {}, {}, {}
    for sa in (1, -1):
        for flip in (1, -1, 0):
            sb = flip * r1 * sa
            q = M.run(U(), W, ev(sa, sb), STEPS, bkind=bk)
            C[(sa, flip)] = proj(q["prediction"], "C", c0, 40)
            res[(sa, flip)] = float(np.max(np.abs(q["residue"][b0:b0 + 30])))
            bsp[(sa, flip)] = spikes_in(q, "B", 1) + spikes_in(q, "B", -1)
    for name, flip in (("agree", 1), ("contradict", -1)):
        # C along what the SEEN B says about C: r2 * sB
        out[name + "_C"] = float(np.mean([C[(sa, flip)] * r2 * flip * r1 * sa for sa in (1, -1)]))
        out[name + "_residue"] = float(np.mean([res[(sa, flip)] for sa in (1, -1)]))
        out[name + "_B_spikes"] = float(np.mean([bsp[(sa, flip)] for sa in (1, -1)]))
    out["open_alone_C"] = float((C[(1, 0)] - C[(-1, 0)]) / 2 * r1 * r2)
    # screening: A's own say over C with B in view (exact answer 0 in a clean Markov world)
    Cb = {}
    for sa in (1, -1):
        for sb in (1, -1):
            Cb[(sa, sb)] = C[(sa, 1)] if sb == r1 * sa else C[(sa, -1)]
    out["A_given_B"] = float(np.mean([(Cb[(1, sb)] - Cb[(-1, sb)]) / 2 for sb in (1, -1)]) * r1 * r2)
    out["B_influence"] = float(np.mean([(Cb[(sa, 1)] - Cb[(sa, -1)]) / 2 for sa in (1, -1)]) * r2)
    if kind == "residue":
        cc = {sa: proj(M.run(U(), W, ev(sa, 0), STEPS, bkind="residue", eyes="closed")["prediction"], "C", c0, 40)
              for sa in (1, -1)}
        out["closed_residue"] = float((cc[1] - cc[-1]) / 2 * r1 * r2)
        return out
    # step one: the imagined B alone (B units cut), and the body's own size gain G (as relay/)
    bb, pk = {}, {}
    for sa in (1, -1):
        q = M.run(U(), W, ev(sa, 0), STEPS, bkind="content", cut=("B",))
        bb[sa] = proj(q["prediction"], "B", b0, 30)
        pk[sa] = float(np.max(np.abs(q["prediction"][b0 - 5:b0 + 30, M.READ["B"]])))
    seen_pk = float(np.max(np.abs(passive([(M.SITES["B"], b0, 1.0)], STEPS)[:, M.READ["B"]])))
    G = seen_pk / max(np.mean(list(pk.values())), 1e-9)
    out["step_one"] = float((bb[1] - bb[-1]) / 2 * r1)
    out["gain_G"] = G
    # eyes closed, three ways for the B units to take up the imagined B
    for bkind in ("content", "regen", "twoform"):
        cc, sizes, lag = {}, [], []
        for sa in (1, -1):
            q = M.run(U(), W, ev(sa, 0), STEPS, bkind=bkind, eyes="closed", gain=G)
            cc[sa] = proj(q["prediction"], "C", c0, 40)
            right = U().index("B", int(r1 * sa))
            sizes += [q["b_size"][k] for k in right if q["spikes"][k]]
        out["closed_" + bkind] = float((cc[1] - cc[-1]) / 2 * r1 * r2)
        if bkind == "twoform":
            out["twoform_size_median"] = float(np.median(sizes)) if sizes else float("nan")
            out["twoform_units_fired"] = len(sizes) / 2
    return out


def job(arg):
    seed, world, rel, kind = arg
    w = WORLDS[world]
    gap, W = train(seed, w, rel, kind)
    return {"seed": seed, "world": world, "kind": kind, "rho1": rel[0], "rho2": rel[1], "gap": gap,
            "weights": np.round(W, 7).tolist(), **read(W, gap, rel, w, kind)}
