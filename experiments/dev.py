"""Development bodies (seeds 981-986). Not evidence: used only to set up the protocol."""
import json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import onebody as O

if __name__ == "__main__":
    seeds = [981, 982, 983, 984, 985, 986]
    jobs = [(s, wd, O.RELATIONS[i % 4], kind) for wd in O.WORLDS for i, s in enumerate(seeds) for kind in ("content", "residue")]
    t0 = time.time(); out = []
    with ProcessPoolExecutor(2) as pool:
        for r in pool.map(O.job, jobs):
            out.append(r)
    pathlib.Path(__file__).resolve().parent.joinpath("dev_results.json").write_text(json.dumps(
        {"label": "development bodies (not evidence)", "results": out}, indent=1))
    print("done %.0fs" % (time.time() - t0))
