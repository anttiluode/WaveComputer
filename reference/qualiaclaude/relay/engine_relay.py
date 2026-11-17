"""Relay: screen off the shortcut when B is seen, imagine B when it is not.

Joins screen/ (residual learning drops the direct A -> C axon once the real B carries C) and chain/ (an
imagined B relays a clue to C). Same vendored nodes and wave physics as both.

Populations (latched fade-triggered clocks, half-wave rectified, so sign selective):
  AB   hear A (A's receptor, aperture 0); fire around B's time; branches write at sites 3-5 (B's place)
  AC   hear A; fire around C's time; branches write at sites 6-7 (C's place): the direct shortcut
  B    EYES OPEN: hear the real B (B's receptor, aperture 4). EYES CLOSED: hear the prediction field at
       aperture 4, i.e. the B that the AB branches imagine. Branches write at sites 6-7.
Targets: at sites 3-5 B's own wave, at 6-7 C's own wave. Learning happens only with eyes open.

Eyes closed, two ways for B-units to take up the imagined B (set in imagine=...):
  "normalized"  the imagined B is rescaled by a body-level gain G to the size of a perceived B, and every
                B-unit keeps its perception threshold (0.8 of its calibration peak). All right-sign units
                fire as if B were seen: the relay passes B's SIGN, not how sure the body is.
  "graded"      the same rescaled drive keeps the clocks on time, but unit k may only arm if the RAW
                imagined B reached theta_k of a perceived B, with theta_k spread uniformly over (0, 1). The
                fraction of B-units that wake is then about the imagined B's strength, so the C
                prediction is about strength(B | A) x weight(C | B): the product the exact answer needs.
G is measured by the body on itself (peak of a perceived B over peak of its own imagined B, with the
B-units silent), so nothing about the answer is given to it.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import engine as E

SITES = {"A": 0, "B": 4, "C": 7}
READ = {"A": 0, "B": 4}
REGION = {"AB": (3, 4, 5), "AC": (6, 7), "B": (6, 7)}
HEARS = {"AB": "A", "AC": "A", "B": "B"}
TIP = {"B": (3, 4, 5), "C": (6, 7)}
REFRACTORY = 300
N = E.N_SITES
K_LEN = 12


class Unit:
    def __init__(self, decay, th, pop, half, theta=0.8):
        self.decay, self.th, self.pop, self.half, self.theta = decay, th, pop, half, theta


_KERNEL = None
def kernel():
    global _KERNEL
    if _KERNEL is None:
        K = np.zeros((N, K_LEN, N))
        for sw in range(3, 8):
            f = E.make("WaveFieldNode", E.WAVE)
            for j in range(K_LEN):
                f.pre_step()
                if j == 0:
                    f.set_input("write_%d" % sw, 1.0)
                f.step(); K[sw, j] = f.read_values
        _KERNEL = K
    return _KERNEL


def run(units, W, events, steps, learn=False, rule=None, mode="residual", eyes="open", imagine="normalized",
        gain=1.0, cut=(), record=False):
    rule = rule or {}
    tb = E.passive([ev for ev in events if ev[0] == SITES["B"]], steps)
    tc = E.passive([ev for ev in events if ev[0] == SITES["C"]], steps)
    target = np.zeros_like(tb); target[:, 3:6] = tb[:, 3:6]; target[:, 6:8] = tc[:, 6:8]
    sensor = {"A": E.passive([ev for ev in events if ev[0] == SITES["A"]], steps)[:, READ["A"]], "B": tb[:, READ["B"]]}
    K = kernel()
    pred = E.make("WaveFieldNode", E.WAVE)
    us = []
    for u in units:
        arm = u.th[0]
        if u.pop == "B" and eyes == "closed":
            p = u.th[0] / 0.8                                 # the unit's calibration peak
            # drive is gain x imagined B, so "raw imagined B reached theta of a perceived B" is arm = theta*gain*p
            arm = u.theta * gain * p if imagine == "graded" else 0.8 * p
        us.append(E.make("FadingHistorySpikeNode", {"decay": u.decay, "arm_threshold": arm, "release_threshold": u.th[1],
                                                    "pulse_amplitude": E.PING_AMP, "refractory": REFRACTORY, "enabled": 1}))
    gs = [E.make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=int(learn))) for w in W]
    closed = [False] * len(units); last = [0.0] * len(units)
    fire_tick = [None] * len(units); fire_w = [None] * len(units)
    out_wave, spikes, heard = [], [[] for _ in units], []
    for tick in range(steps):
        prev_pred = np.array([pred.get_output("read_%d" % s) for s in range(N)])
        prev_target = target[tick - 1] if tick > 0 else np.zeros(N)
        prev_pulse = [u.get_output("pulse") for u in us]
        prev_out = [[g.get_output("out_%d" % s) for s in range(N)] for g in gs]
        pred.pre_step()
        for n in us + gs:
            n.pre_step()
        for k, (u, spec) in enumerate(zip(us, units)):
            if tick == 0:
                src = 0.0
            elif spec.pop == "B" and eyes == "closed":
                src = gain * float(prev_pred[READ["B"]])
            else:
                src = float(sensor[HEARS[spec.pop]][tick - 1])
            x = max(0.0, spec.half * src)
            u.set_input("signal_in", 0.0 if closed[k] else x)
        for k, (g, p, o, spec) in enumerate(zip(gs, prev_pulse, prev_out, units)):
            g.set_input("pre_spike", p)
            region = REGION[spec.pop]
            if mode == "residual":
                tip = prev_pred
            else:
                tip = np.zeros(N)
                if fire_tick[k] is not None:
                    j = tick - fire_tick[k] - 3
                    if 0 <= j < K_LEN:
                        tip = E.PING_AMP * sum(fire_w[k][sw] * K[sw, j] for sw in region)
            for s in range(N):
                ok = s in region
                g.set_input("site_%d" % s, float(tip[s] - prev_target[s]) if ok else 0.0)
                if ok and o[s] and spec.pop not in cut:
                    pred.set_input("write_%d" % s, o[s])
        pred.step()
        for n in us + gs:
            n.step()
        for k, u in enumerate(us):
            if u.pulse:
                spikes[k].append(tick); closed[k] = False
                fire_tick[k] = tick; fire_w[k] = list(gs[k].weights)
            elif u.armed and u.trace < last[k]:
                closed[k] = True
            last[k] = u.trace
        out_wave.append(pred.read_values.copy())
    weights = [[(w if s in REGION[spec.pop] else 0.0) for s, w in enumerate(g.weights)] for g, spec in zip(gs, units)]
    return {"prediction": np.asarray(out_wave, dtype=float), "weights": weights, "spikes": spikes}
