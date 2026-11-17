"""The pre-registered batch (see PROTOCOL.md). About 12 minutes on two cores.

    python experiments/run.py            writes experiments/results.json (with hashes of protocol, runner, engine)
"""
from __future__ import annotations
import hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import onebody as O

SEEDS = list(range(200001, 200033))


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    jobs = [(s, wd, O.RELATIONS[i % 4], kind) for wd in O.WORLDS for i, s in enumerate(SEEDS) for kind in ("content", "residue")]
    receipt = {"label": "WaveComputer one-body batch (pre-registered)", "worlds": O.WORLDS, "seeds": SEEDS,
               "hashes": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "PROTOCOL.md", HERE / "run.py", HERE / "onebody.py",
                          ROOT / "wavecomputer/machine.py", ROOT / "wavecomputer/nodes.py", ROOT / "wavecomputer/physics.py",
                          ROOT / "wavecomputer/units.json", ROOT / "wavecomputer/kernel.npz"]},
               "results": []}
    t0 = time.time()
    with ProcessPoolExecutor(2) as pool:
        for n, r in enumerate(pool.map(O.job, jobs), 1):
            receipt["results"].append(r); receipt["elapsed_seconds"] = time.time() - t0
            if n % 16 == 0 or n == len(jobs):
                (HERE / "results.json").write_text(json.dumps(receipt, indent=1))
                print("%d/%d %.0fs" % (n, len(jobs), time.time() - t0), flush=True)
