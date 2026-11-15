"""Build the v4 gym dataset: python build_gym.py OUT_DIR [N_CLEAN N_SINGLE_EACH N_COMBO] [SEED0]

Default 15 clean + 3 x 6 single faults + 15 combined = 48 envs. Task ids gym-NNN are in a seeded random order,
so the id says nothing about the fault. tasks.jsonl (next to this script) maps id -> seed, faults, reference accuracies.
"""
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import gen  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/work/vals-mouse/valkyrie/dataset-broken-v4")
NC, NS, NK = (int(v) for v in sys.argv[2:5]) if len(sys.argv) > 4 else (15, 3, 15)
SEED0 = int(sys.argv[5]) if len(sys.argv) > 5 else 4000

specs = [[] for _ in range(NC)] + [[f] for f in gen.ALL_FAULTS for _ in range(NS)] + [list(gen.COMBOS[i % len(gen.COMBOS)]) for i in range(NK)]
rng = np.random.default_rng(SEED0)
order = rng.permutation(len(specs))
if __name__ == "__main__":
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)


def one(job):
    j, i = job
    seed = SEED0 + int(i)
    tid = f"gym-{j + 1:03d}"
    env, key, st = gen.generate(seed, specs[i], OUT, tid)
    r = dict(task_id=tid, seed=seed, faults=env["faults"], kind=env["kind"], stim=env["info"]["stim"], K=env["info"]["K"],
             U=env["info"]["U"], n=[len(env["train"][1]), len(env["val"][1]), len(env["test_y"])], fmt=env["fmt"],
             ref_acc=key["ref_acc"], naive_cv=key["naive_cv"], naive_acc=key["naive_acc"], aware_acc=key["aware_acc"],
             fix_applies=key["fix_applies"], fix_target=key["fix_target"])
    print(json.dumps(r), flush=True)
    return r


if __name__ == "__main__":
    with Pool(int(os.environ.get("GYM_PROCS", "8"))) as pool:
        rows = pool.map(one, list(enumerate(order)), chunksize=1)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks_v4.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
