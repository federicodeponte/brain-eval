# stream trials + spike counts for one DANDI 000004 session -> npz
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])) if False else sys.argv[3])
from common import open_remote
path, out = sys.argv[1], sys.argv[2]
f = open_remote("000004", path)
T = f["intervals/trials"]
def s(x): return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x[:]])
on = T["stim_on_time"][:]; off = T["stim_off_time"][:]
cat = s(T["category_name"]); phase = s(T["stim_phase"]); no = s(T["new_old_labels_recog"])
img = s(T["external_image_file"])
st = f["units/spike_times"][:]; idx = f["units/spike_times_index"][:]
starts = np.r_[0, idx[:-1]]
el = f["units/electrodes"][:]
try:
    loc = s(f["general/extracellular_ephys/electrodes/location"]); uloc = loc[el]
except Exception as e:
    uloc = np.array(["?"] * len(idx))
wins = {"w200_1200": (0.2, 1.2), "w0_1000": (0.0, 1.0), "w200_1700": (0.2, 1.7)}
res = {}
for k, (a, b) in wins.items():
    C = np.zeros((len(on), len(idx)), np.int32)
    for u in range(len(idx)):
        sp = st[starts[u]:idx[u]]
        C[:, u] = np.searchsorted(sp, on + b) - np.searchsorted(sp, on + a)
    res[k] = C
base = np.zeros((len(on), len(idx)), np.int32)
for u in range(len(idx)):
    sp = st[starts[u]:idx[u]]
    base[:, u] = np.searchsorted(sp, on) - np.searchsorted(sp, on - 1.0)
np.savez_compressed(out, cat=cat, phase=phase, newold=no, img=img, on=on, off=off, uloc=uloc, base=base, path=path, **res)
print(path, len(on), len(idx), sorted(set(cat)), sorted(set(phase)), sorted(set(uloc))[:6])
