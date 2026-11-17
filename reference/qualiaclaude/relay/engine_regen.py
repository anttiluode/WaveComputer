"""Relay follow-up: give the imagined B the SHAPE IN TIME of a seen B (relay/engine_relay.py is not modified).

relay/ found that an imagined B wakes the B-units 10 to 23 frames late in most bodies: it is a wider, softer
wave than a seen B, and a fade-triggered clock waits for its tail. Two eyes-closed readouts are added:

  regen      a regenerative event at B's place. The B-units hear the imagined B (gain-scaled, as in
             "normalized") until it first crosses half the peak of a seen B, in either sign. From that tick
             on they hear the rest of a canonical seen-B wave of that sign, from its own half-peak point
             onward. It is causal: nothing before the crossing is changed. This is the picture of a local
             dendritic spike taking over from a soft input: once the threshold is crossed, the event has its
             own fixed shape.
  regen_amp  the same, and the B-units' outgoing ping is scaled by the body's own estimate of how sure it is
             of B: |beta_B|, the projection of its own imagined B on a seen B. The clocks are untouched; only
             the payload changes, so C ~ (how sure of B) x (what B says about C).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import engine as E
import engine_relay as L

N = E.N_SITES
CROSS = 0.5                      # fraction of a seen B's peak; fixed before any run


def canonical(steps):
    """A seen B of amplitude 1 at aperture 4, its peak, and the tick (from onset) it first reaches CROSS x peak."""
    onset = 20
    w = E.passive([(L.SITES["B"], onset, 1.0)], steps + 200)[:, L.READ["B"]]
    pk = float(np.max(np.abs(w)))
    t_half = int(np.argmax(np.abs(w) >= CROSS * pk))
    return w[t_half:], pk, t_half - onset


def run(units, W, events, steps, gain=1.0, ping_scale=1.0):
    """Eyes closed, regenerative imagined B. Returns prediction, spikes and the regenerated onset."""
    tail, pk, d0 = canonical(steps)
    sensor_a = E.passive([ev for ev in events if ev[0] == L.SITES["A"]], steps)[:, L.READ["A"]]
    pred = E.make("WaveFieldNode", E.WAVE)
    us = [E.make("FadingHistorySpikeNode", {"decay": u.decay, "arm_threshold": u.th[0], "release_threshold": u.th[1],
                                            "pulse_amplitude": E.PING_AMP * (ping_scale if u.pop == "B" else 1.0),
                                            "refractory": L.REFRACTORY, "enabled": 1}) for u in units]
    gs = [E.make("DampingReturnNode", dict(weights=list(w), learn_enabled=0)) for w in W]
    closed = [False] * len(units); last = [0.0] * len(units)
    spikes = [[] for _ in units]; out_wave = []
    t_cross, sign = None, 0
    for tick in range(steps):
        prev_pred = np.array([pred.get_output("read_%d" % s) for s in range(N)])
        prev_pulse = [u.get_output("pulse") for u in us]
        prev_out = [[g.get_output("out_%d" % s) for s in range(N)] for g in gs]
        pred.pre_step()
        for n in us + gs:
            n.pre_step()
        imagined = gain * float(prev_pred[L.READ["B"]]) if tick > 0 else 0.0
        if t_cross is None and abs(imagined) >= CROSS * pk:
            t_cross, sign = tick, (1 if imagined > 0 else -1)
        b_in = imagined if t_cross is None else sign * float(tail[tick - t_cross]) if tick - t_cross < len(tail) else 0.0
        for k, (u, spec) in enumerate(zip(us, units)):
            src = 0.0 if tick == 0 else (b_in if spec.pop == "B" else float(sensor_a[tick - 1]))
            x = max(0.0, spec.half * src)
            u.set_input("signal_in", 0.0 if closed[k] else x)
        for k, (g, p, o, spec) in enumerate(zip(gs, prev_pulse, prev_out, units)):
            g.set_input("pre_spike", p)
            for s in L.REGION[spec.pop]:
                if o[s]:
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
    onset = (t_cross - 1 - d0) if t_cross is not None else None    # the seen-B onset this regenerated event matches
    return {"prediction": np.asarray(out_wave, dtype=float), "spikes": spikes, "regen_onset": onset, "regen_sign": sign}
