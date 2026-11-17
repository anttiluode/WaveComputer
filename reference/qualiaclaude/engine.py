"""Engine for QualiaClaude: one shared wave field, two senses, three banks of fade-triggered units.

Sense 1, the wave tank: stimuli are two-frame pulses written into the vendored upstream WaveFieldNode
(96x96) at sites A = 0, B = 7, C = 4, as in StateShapedPingRecall and PerceptionLabAI.
Sense 2, a tone line: a separate channel, never written into the field. Tone 1 and tone 2 are
two-frame pulses on two separate lines.

Units are FadingHistorySpikeNodes. Each hears ONE input and never the body's own output:
  A-bank     24 timers, hear the world's wave at aperture 1 (the passive field of the stimuli; as in
             StateShapedPingRecall damping2, "world" input, so the body cannot move its own clock)
  tone banks 12 timers per tone, hear their tone line
Every unit returns into the shared field through a DampingReturnNode (eight signed branches, one per
site) that learns to flatten the field at its own tips after its unit fires. The shared field is the
only place the banks meet: it carries the population's joint prediction.

Host step order as in PerceptionLab: read previous outputs, set inputs, step every node.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "tests"))
from host_shim import load

N_SITES = 8
SITES = {"A": 0, "B": 7, "C": 4}
A_READ = 1
STIM_AMP, PING_AMP, TONE_AMP = 0.12, 0.24, 1.0
WAVE = {"size": 96, "speed": 0.5, "damping": 0.006, "substeps": 4, "drive_gain": 0.6, "source_radius": 4,
        "edge_absorb": 0.06, "read_aperture": 6, "line_y": 0.5, "first_x": 0.2, "last_x": 0.92, "source_x": 0.08}
# A-bank: the "even" fade rates of StateShapedPingRecall damping2 (fire ~4 ticks apart, ticks 51-150 after A at 23)
A_DECAYS = [0.756, 0.8405, 0.891, 0.9095, 0.921, 0.9225, 0.923, 0.925, 0.928, 0.933, 0.9405, 0.9455,
            0.949, 0.9515, 0.956, 0.961, 0.9615, 0.962, 0.9625, 0.9635, 0.964, 0.965, 0.9665, 0.9695]

_cls = {}
def cls(name):
    if name not in _cls:
        path = {"WaveFieldNode": ROOT / "vendor/up/nodes/wavefieldnode.py",
                "FadingHistorySpikeNode": ROOT / "vendor/nodes/fading_history_spike_node.py",
                "DampingReturnNode": ROOT / "vendor/nodes/damping_return_node.py"}[name]
        _cls[name] = load(path, name)
    return _cls[name]


def make(name, cfg):
    n = cls(name)()
    if hasattr(n, "set_config_options"):
        n.set_config_options(dict(cfg))
    for k, v in cfg.items():
        if hasattr(n, k) and getattr(n, k) != v:
            setattr(n, k, v)
    n._render = lambda *a: None        # drawing only
    return n


def passive(stimuli, steps):
    """Sensor waves (steps x 8) of the field driven by the stimuli alone."""
    f = make("WaveFieldNode", WAVE); rows = []
    for tick in range(steps):
        f.pre_step()
        for site, onset, scale in stimuli:
            if onset + 1 <= tick < onset + 3:
                f.set_input("write_%d" % site, STIM_AMP * scale)
        f.step(); rows.append(f.read_values.copy())
    return np.asarray(rows, dtype=float)


def tone_line(onset, steps):
    x = np.zeros(steps)
    if onset is not None:
        x[onset + 1:onset + 3] = TONE_AMP
    return x


def peak_trace(decay, signal):
    tr = peak = 0.0
    for v in signal:
        tr = decay * tr + (1 - decay) * abs(v); peak = max(peak, tr)
    return peak


def thresholds(decay, signal, arm=0.80, release=0.20):
    """Label-blind: arm and release as fractions of the unit's own peak trace under its calibration input."""
    p = peak_trace(decay, signal)
    return (arm * p, release * p)


class Body:
    """Unit specification: list of (decay, (arm, release), source) with source ("wave", aperture) or ("tone", k)."""
    def __init__(self, units):
        self.units = units


def run(body, W, stimuli, tones, steps, learn=False, rule=None, return_spikes=True):
    """stimuli: [(site, onset, scale)] into the field. tones: {k: onset or None}.
    W: one list of 8 signed weights per unit. Returns sensor waves, final weights, spikes, clipped frames."""
    rule = rule or {}
    heard_wave = passive(stimuli, steps)
    lines = {k: tone_line(on, steps) for k, on in tones.items()}
    field = make("WaveFieldNode", WAVE)
    units = [make("FadingHistorySpikeNode", {"decay": d, "arm_threshold": th[0], "release_threshold": th[1],
                                             "pulse_amplitude": PING_AMP, "refractory": 3, "enabled": 1})
             for d, th, _ in body.units]
    grows = [make("DampingReturnNode", dict(rule, weights=list(w), learn_enabled=int(learn))) for w in W]
    nodes = [field] + units + grows
    wave, spikes, clip = [], [[] for _ in units], 0
    for tick in range(steps):
        prev_read = [field.get_output("read_%d" % s) for s in range(N_SITES)]
        prev_pulse = [u.get_output("pulse") for u in units]
        prev_out = [[g.get_output("out_%d" % s) for s in range(N_SITES)] for g in grows]
        for n in nodes:
            n.pre_step()
        for u, (_, _, src) in zip(units, body.units):
            if src[0] == "wave":
                v = float(heard_wave[tick - 1, src[1]]) if tick > 0 else 0.0
            else:
                line = lines.get(src[1])
                v = float(line[tick - 1]) if (line is not None and tick > 0) else 0.0
            u.set_input("signal_in", v)
        for g, p, o in zip(grows, prev_pulse, prev_out):
            g.set_input("pre_spike", p)
            for s in range(N_SITES):
                g.set_input("site_%d" % s, prev_read[s])
                if o[s]:
                    field.set_input("write_%d" % s, o[s])
        for site, onset, scale in stimuli:
            if onset + 1 <= tick < onset + 3:
                field.set_input("write_%d" % site, STIM_AMP * scale)
        for n in nodes:
            n.step()
        for k, u in enumerate(units):
            if u.pulse:
                spikes[k].append(tick)
        wave.append(field.read_values.copy())
        clip += int(float(np.max(np.abs(field.field))) >= 3.9998 or float(np.max(np.abs(field.vel))) >= 3.9998)
    return {"wave": np.asarray(wave, dtype=float), "weights": [list(map(float, g.weights)) for g in grows],
            "spikes": spikes, "clipped": clip}
