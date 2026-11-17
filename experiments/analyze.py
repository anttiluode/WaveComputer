"""Score the batch against the rules of PROTOCOL.md. Writes experiments/summary.json.

    python experiments/analyze.py [results.json]
"""
from __future__ import annotations
import json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import onebody as O


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    by = {}
    for r in d["results"]:
        by[(r["world"], r["kind"], r["seed"])] = r
    seeds = sorted({r["seed"] for r in d["results"]})
    return d, by, seeds


def med(x):
    return float(np.median(x))


def summarize(path):
    d, by, seeds = load(path)
    n = len(seeds); need = math.ceil(0.9 * n); need_pair = math.ceil(0.75 * n)
    g = lambda wd, kind, key: np.array([by[(wd, kind, s)][key] for s in seeds])
    B = {wd: O.bayes(w) for wd, w in O.WORLDS.items()}
    S = {"n": n, "need_90pct": need, "need_75pct": need_pair, "rules": {}, "report": {}}
    R = S["rules"]

    # P1 perception, same body: a seen B settles C, whether it agrees with A or contradicts it
    for wd in ("clean85", "clean70"):
        ag, co = g(wd, "content", "agree_C"), g(wd, "content", "contradict_C")
        R["P1_" + wd] = {"contradict_right_sign": int((co > 0).sum()), "contradict_median_of_bayes": med(co) / B[wd]["contradict"],
                         "agree_median_of_bayes": med(ag) / B[wd]["agree"],
                         "pass": bool((co > 0).sum() >= need and med(co) >= 0.5 * B[wd]["contradict"])}
    # P2 screening: with B in view, A's own say over C is about zero
    for wd in ("clean85", "clean70"):
        a = g(wd, "content", "A_given_B")
        R["P2_" + wd] = {"median_abs": med(np.abs(a)), "median": med(a), "pass": bool(med(np.abs(a)) <= 0.15)}
    # P3 imagination chains A -> imagined B -> C, eyes closed, two-form unit
    for wd in ("clean85", "clean70"):
        c = g(wd, "content", "closed_twoform")
        R["P3_" + wd] = {"right_sign": int((c > 0).sum()), "median_of_bayes": med(c) / B[wd]["closed"],
                         "pass": bool((c > 0).sum() >= need)}
    # P4 imagination follows the reliability of A -> B (paired seeds: same schedule, only q1 changed)
    bayes_ratio = B["clean70"]["closed"] / B["clean85"]["closed"]
    ratio = {}
    for kind, key in (("content", "closed_twoform"), ("content", "closed_regen"), ("content", "closed_content"), ("residue", "closed_residue")):
        hi, lo = g("clean85", kind, key), g("clean70", kind, key)
        r = np.where(hi > 0, lo / np.where(hi > 0, hi, 1), np.nan)      # undefined when the 0.85 body has the wrong sign
        ratio[key] = r
        S["report"]["ratio_" + key] = {"median": float(np.nanmedian(r)), "undefined": int(np.isnan(r).sum()), "per_seed": r.tolist()}
    err = lambda r: np.where(np.isnan(r) | (r <= 0), np.inf, np.abs(np.log(np.where(r > 0, r, 1) / bayes_ratio)))
    closer = int((err(ratio["closed_twoform"]) < err(ratio["closed_regen"])).sum())
    m = float(np.nanmedian(ratio["closed_twoform"]))
    R["P4"] = {"bayes_ratio": bayes_ratio, "twoform_ratio_median": m, "regen_ratio_median": float(np.nanmedian(ratio["closed_regen"])),
               "twoform_closer_than_regen": closer, "pass": bool(0.40 <= m <= 0.80 and closer >= need_pair)}
    # P5 the residue is held as well: quiet when B agrees, loud when it contradicts
    for wd in ("clean85", "clean70"):
        a, c = g(wd, "content", "agree_residue"), g(wd, "content", "contradict_residue")
        R["P5_" + wd] = {"louder_on_contradiction": int((c > a).sum()), "agree_median": med(a), "contradict_median": med(c),
                         "pass": bool((c > a).sum() >= need)}
    # N1, N2 precision weighting from learning alone (QualeX's open test): noisy B, no gain anywhere
    cc, cn = g("clean85", "content", "contradict_C"), g("noisy85", "content", "contradict_C")
    R["N1"] = {"noisy_contradict_median_abs": med(np.abs(cn)), "noisy_contradict_median": med(cn), "clean_contradict_median": med(cc),
               "clean_above_noisy": int((cc > cn).sum()),
               "pass": bool(med(np.abs(cn)) <= 0.15 and med(cc) >= 0.35 and (cc > cn).sum() >= need)}
    ac, an = g("clean85", "content", "A_given_B"), g("noisy85", "content", "A_given_B")
    R["N2"] = {"noisy_A_given_B_median": med(an), "clean_A_given_B_median": med(ac), "noisy_above_clean": int((an > ac).sum()),
               "bayes_noisy": B["noisy85"]["k"] * 0.5 * (B["noisy85"]["agree"] / B["noisy85"]["k"] - B["noisy85"]["contradict"] / B["noisy85"]["k"]),
               "pass": bool(0.15 <= med(an) <= 0.5 and (an > ac).sum() >= need)}

    # reported without a rule
    rep = S["report"]
    for wd in O.WORLDS:
        row = {"bayes": B[wd]}
        for kind, key in (("content", "closed_content"), ("content", "closed_regen"), ("content", "closed_twoform"), ("residue", "closed_residue")):
            c = g(wd, kind, key)
            row[key] = {"median_of_bayes": med(c) / B[wd]["closed"], "right_sign": int((c > 0).sum()),
                        "median_abs_log_error": med(np.abs(np.log(np.clip(c, 1e-6, None) / B[wd]["closed"])))}
        for kind in ("content", "residue"):
            for key in ("agree_C", "contradict_C", "A_given_B", "B_influence", "agree_B_spikes", "contradict_B_spikes", "agree_residue", "contradict_residue"):
                row["%s_%s" % (kind, key)] = med(g(wd, kind, key))
            row[kind + "_contradict_right_sign"] = int((g(wd, kind, "contradict_C") > 0).sum())
        for key in ("step_one", "gain_G", "twoform_size_median"):
            row[key] = med(g(wd, "content", key))
        own = g(wd, "content", "step_one") * g(wd, "content", "B_influence")
        rel = g(wd, "content", "closed_twoform") / np.where(np.abs(own) > 1e-9, own, np.nan)
        row["twoform_vs_own_links_median"] = float(np.nanmedian(rel))
        rel_r = g(wd, "content", "closed_regen") / np.where(np.abs(own) > 1e-9, own, np.nan)
        row["regen_vs_own_links_median"] = float(np.nanmedian(rel_r))
        rep[wd] = row
    S["all_rules_pass"] = all(v["pass"] for v in R.values())
    return S


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json"
    S = summarize(path)
    out = HERE / ("summary.json" if pathlib.Path(path).name == "results.json" else "dev_summary.json")
    out.write_text(json.dumps(S, indent=1))
    for k, v in S["rules"].items():
        print("%-12s %s  %s" % (k, "PASS" if v["pass"] else "FAIL", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "pass"}))
    print("all rules pass:", S["all_rules_pass"])
