"""The figure for the README: experiments/onebody.png, from experiments/results.json."""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import onebody as O

INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COL = {"content": "#8c959f", "regen": "#cf6a2e", "twoform": "#1a7f64", "residue": "#5b6cc4"}
NAME = {"content": "content (relay)", "regen": "regen (certain)", "twoform": "two-form", "residue": "residue (innovation)"}


def main(path=HERE / "results.json", out=HERE / "onebody.png"):
    d = json.loads(pathlib.Path(path).read_text())
    by = {(r["world"], r["kind"], r["seed"]): r for r in d["results"]}
    seeds = sorted({r["seed"] for r in d["results"]})
    g = lambda wd, kind, key: np.array([by[(wd, kind, s)][key] for s in seeds])
    B = {wd: O.bayes(w) for wd, w in O.WORLDS.items()}
    rng = np.random.default_rng(0)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1.2]})

    # 1. imagination, eyes closed, as a fraction of Bayes
    a = ax[0]; kinds = [("content", "closed_content"), ("regen", "closed_regen"), ("twoform", "closed_twoform"), ("residue", "closed_residue")]
    for j, wd in enumerate(("clean85", "clean70")):
        for i, (k, key) in enumerate(kinds):
            v = g(wd, "residue" if k == "residue" else "content", key) / B[wd]["closed"]
            x = j * 5 + i
            a.scatter(x + rng.uniform(-.18, .18, len(v)), np.clip(v, -1, 4), s=7, color=COL[k], alpha=.45, lw=0)
            a.plot([x - .3, x + .3], [np.median(v)] * 2, color=COL[k], lw=2.5)
    a.axhline(1, color=INK, lw=.8, ls="--"); a.axhline(0, color=GRID, lw=.8)
    a.set_xticks([1.5, 6.5]); a.set_xticklabels(["A → B reliable (q₁ 0.85)", "A → B weaker (q₁ 0.70)"])
    a.set_ylabel("eyes closed, A alone: C / Bayes"); a.set_ylim(-1, 4)
    a.set_title("Imagination: how sure is C?", loc="left", color=INK, fontsize=10)
    for k in ("content", "regen", "twoform", "residue"):
        a.plot([], [], color=COL[k], lw=2.5, label=NAME[k])
    a.legend(frameon=False, fontsize=8, loc="upper left")

    # 2. paired ratio: lower the reliability of A -> B, same seed
    a = ax[1]; br = B["clean70"]["closed"] / B["clean85"]["closed"]
    for i, (k, key) in enumerate([("regen", "closed_regen"), ("twoform", "closed_twoform"), ("residue", "closed_residue")]):
        kind = "residue" if k == "residue" else "content"
        hi, lo = g("clean85", kind, key), g("clean70", kind, key)
        r = np.where(hi > 0, lo / np.where(hi > 0, hi, 1), np.nan)
        a.scatter(i + rng.uniform(-.18, .18, len(r)), np.clip(r, 0, 2.5), s=7, color=COL[k], alpha=.45, lw=0)
        a.plot([i - .3, i + .3], [np.nanmedian(r)] * 2, color=COL[k], lw=2.5)
    a.axhline(br, color=INK, lw=.8, ls="--"); a.text(-0.55, br - 0.05, "Bayes %.2f" % br, va="top", ha="left", fontsize=8, color=INK)
    a.axhline(1, color=GRID, lw=.8)
    a.set_xticks(range(3)); a.set_xticklabels(["regen", "two-form", "residue"])
    a.set_ylabel("eyes-closed C, q₁ 0.70 / q₁ 0.85 (same seed)"); a.set_ylim(0, 2.5); a.set_xlim(-.6, 3.2)
    a.set_title("Does imagination follow reliability?", loc="left", color=INK, fontsize=10)

    # 3. perception: a seen B that contradicts A, clean and noisy
    a = ax[2]
    rows = [("clean85", "content"), ("clean85", "residue"), ("noisy85", "content"), ("noisy85", "residue")]
    for i, (wd, kind) in enumerate(rows):
        v = g(wd, kind, "contradict_C")
        k = "twoform" if kind == "content" else "residue"
        a.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=7, color=COL[k], alpha=.45, lw=0)
        a.plot([i - .3, i + .3], [np.median(v)] * 2, color=COL[k], lw=2.5)
        a.plot([i - .38, i + .38], [B[wd]["contradict"]] * 2, color=INK, lw=.8, ls="--")
    a.axhline(0, color=GRID, lw=.8)
    a.set_xticks(range(4)); a.set_xticklabels(["clean\ntwo-form", "clean\nresidue", "noisy B\ntwo-form", "noisy B\nresidue"])
    a.set_ylabel("eyes open, B contradicts A: C along B"); a.set_title("Perception: does a contradiction win?", loc="left", color=INK, fontsize=10)
    fig.text(0.995, 0.01, "dots: bodies · bars: medians · dashed: Bayes", ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1], HERE / "dev_onebody.png")
    else:
        main()
