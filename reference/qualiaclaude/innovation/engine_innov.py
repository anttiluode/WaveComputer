"""Innovation body: the dendrite keeps what it expects; only the residue reaches the soma.

Antti's rule (16 Nov 2026): what a neuron already expects -- its own imagined B, its own state coming back --
cancels in the dendrite and never reaches the soma. Only the residue spills over and becomes a spike.
In numbers:  B = B_hat(A) + r_B.  The held prediction B_hat is state; the spike carries only r_B.
This is the innovation form of a Kalman filter / predictive coding (Rao & Ballard 1999).

Same populations, nodes and wave physics as relay/ (latched fade-triggered clocks, separate receptors):
  AB   hear A; write at sites 3-5 (B's place): the held prediction of B, B_hat, in the prediction field
  AC   hear A; write at sites 6-7 (C's place): the PRIOR path to C. It is legitimate here, not a shortcut:
       B-units no longer carry all of B, only what A did not predict
  B    EYES OPEN: hear the residue  r_B = (B's wave at aperture 4) - (prediction field at aperture 4).
       Both are the same physical quantity (the AB branches learn to match B's wave there), so no gain.
       EYES CLOSED: the senses are off, so there is no residue; B-units hear nothing.
Targets: B's own wave at 3-5, C's own wave at 6-7. Residual learning rule (DampingReturnNode), eyes open.
Eyes closed, the exact answer is E[C|A] = (2q1-1)(2q2-1) A, and here it can only come through AC.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "relay"))
import engine as E
import engine_relay as L

N = E.N_SITES


def run(units, W, events, steps, learn=False, rule=None, eyes="open", b_arm=0.8, cut=()):
    rule = rule or {}
    tb = E.passive([ev for ev in events if ev[0] == L.SITES["B"]], steps)
    tc = E.passive([ev for ev in events if ev[0] == L.SITES["C"]], steps)
    target = np.zeros_like(tb); target[:, 3:6] = tb[:, 3:6]; target[:, 6:8] = tc[:, 6:8]
    sensor_a = E.passive([ev for ev in events if ev[0] == L.SITES["A"]], steps)[:, L.READ["A"]]
    sensor_b = tb[:, L.READ["B"]]
    pred = E.make("WaveFieldNode", E.WAVE)
    us = []
    for u in units:
        arm = (b_arm * u.th[0] / 0.8) if u.pop == "B" else u.th[0]       # th[0] = 0.8 x calibration peak
        us.append(E.make("FadingHistorySpikeNode", {"decay": u.decay, "arm_threshold": arm, "release_threshold": u.th[1],
                                                    "pulse_amplitude": E.PING_AMP, "refractory": L.REFRACTORY, "enabled": 1}))
    gs = [E.make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=int(learn))) for w in W]
    closed = [False] * len(units); last = [0.0] * len(units)
    spikes = [[] for _ in units]; out_wave = []; resid = []
    for tick in range(steps):
        prev_pred = np.array([pred.get_output("read_%d" % s) for s in range(N)])
        prev_target = target[tick - 1] if tick > 0 else np.zeros(N)
        prev_pulse = [u.get_output("pulse") for u in us]
        prev_out = [[g.get_output("out_%d" % s) for s in range(N)] for g in gs]
        pred.pre_step()
        for n in us + gs:
            n.pre_step()
        r_b = (float(sensor_b[tick - 1]) - float(prev_pred[L.READ["B"]])) if (tick > 0 and eyes == "open") else 0.0
        resid.append(r_b)
        for k, (u, spec) in enumerate(zip(us, units)):
            src = 0.0 if tick == 0 else (r_b if spec.pop == "B" else float(sensor_a[tick - 1]))
            x = max(0.0, spec.half * src)
            u.set_input("signal_in", 0.0 if closed[k] else x)
        for k, (g, p, o, spec) in enumerate(zip(gs, prev_pulse, prev_out, units)):
            g.set_input("pre_spike", p)
            region = L.REGION[spec.pop]
            for s in range(N):
                ok = s in region
                g.set_input("site_%d" % s, float(prev_pred[s] - prev_target[s]) if ok else 0.0)
                if ok and o[s] and spec.pop not in cut:
                    pred.set_input("write_%d" % s, o[s])
        pred.step()
        for n in us + gs:
            n.step()
        for k, u in enumerate(us):
            if u.pulse:
                spikes[k].append(tick); closed[k] = False
            elif u.armed and u.trace < last[k]:
                closed[k] = True
            last[k] = u.trace
        out_wave.append(pred.read_values.copy())
    weights = [[(w if s in L.REGION[spec.pop] else 0.0) for s, w in enumerate(g.weights)] for g, spec in zip(gs, units)]
    return {"prediction": np.asarray(out_wave, dtype=float), "weights": weights, "spikes": spikes, "residue": np.asarray(resid)}
