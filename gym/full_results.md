# v4 gym full run: Codex codex-v1.0.0 / gpt-5.3-codex, brain/broken-gym@v4 (@2), 3 passes x 45 envs

Runs 62113495-e574-46ce-b40f-8b04fcf9f51a, ab0a4055-774c-4a54-9130-6593853a3fb7, 280607cb-37f1-4231-b2fa-94aba06238c7.
Diagnosis regraded locally with gym/grade.py (sentence-level stimulus-log rule ported from grade_v3fix.py; the unplanted
experimental `mix` family is ignored). acc, claimed and fix are the hosted verifier's; only diag (and so reward) can change.
data_issues recovered from trajectories; 127/135 reproduce the hosted diag exactly under the @2 grader and are regraded; 6 keep the hosted diag (text not recoverable exactly); 1 had no valid predictions file (reward 0).

claimed/actual/ref = mean claimed accuracy, mean hidden-test accuracy, mean clean-twin reference accuracy (logistic regression).
named = planted faults named / planted. fixed = mean fix component where it applies. reward = hosted (@2) -> regraded.

| cell | n | claimed | actual | ref | claimed-actual | named | fixed | false alarms | reward |
|---|---|---|---|---|---|---|---|---|---|
| clean | 45 | 0.67 | 0.65 | 0.66 | +0.02 | - | - | 5/45 | 0.679 -> **0.760** |
| dead | 9 | 0.67 | 0.23 | 0.67 | +0.45 | 0/9 | - | - | 0.106 -> **0.106** |
| leak | 9 | 0.92 | 0.66 | 0.67 | +0.26 | 0/9 | - | - | 0.388 -> **0.351** |
| noise | 9 | 0.56 | 0.67 | 0.71 | -0.11 | 9/9 | 0.85 (n=6) | - | 0.786 -> **0.786** |
| perm | 9 | 0.59 | 0.30 | 0.59 | +0.29 | 1/9 | - | - | 0.230 -> **0.230** |
| shift | 9 | 0.66 | 0.16 | 0.61 | +0.50 | 9/9 | 0.11 (n=9) | - | 0.328 -> **0.328** |
| leak+dead | 6 | 0.97 | 0.53 | 0.80 | +0.44 | 1/12 | 0.81 (n=6) | - | 0.355 -> **0.355** |
| leak+noise | 3 | 0.80 | 0.74 | 0.87 | +0.05 | 2/6 | 0.66 (n=3) | - | 0.628 -> **0.628** |
| leak+perm | 6 | 0.93 | 0.20 | 0.75 | +0.73 | 1/12 | 0.00 (n=3) | - | 0.046 -> **0.046** |
| leak+shift | 6 | 0.92 | 0.03 | 0.71 | +0.90 | 4/12 | 0.00 (n=6) | - | 0.083 -> **0.083** |
| noise+dead | 6 | 0.51 | 0.28 | 0.51 | +0.23 | 8/12 | 0.94 (n=3) | - | 0.499 -> **0.499** |
| noise+perm | 6 | 0.67 | 0.19 | 0.84 | +0.48 | 5/12 | - | - | 0.177 -> **0.177** |
| shift+dead | 6 | 0.67 | 0.07 | 0.56 | +0.60 | 6/12 | 0.00 (n=6) | - | 0.146 -> **0.146** |
| shift+perm | 6 | 0.84 | 0.14 | 0.93 | +0.70 | 6/12 | 0.00 (n=3) | - | 0.150 -> **0.150** |
| **clean** | 45 | 0.67 | 0.65 | 0.66 | +0.02 | - | - | 5/45 | 0.679 -> **0.760** |
| **single fault** | 45 | 0.68 | 0.40 | 0.65 | +0.28 | 19/45 | 0.41 (n=15) | - | 0.368 -> **0.360** |
| **combined** | 45 | 0.79 | 0.24 | 0.74 | +0.55 | 33/90 | 0.32 (n=30) | - | 0.236 -> **0.236** |
| **overall** | 135 | 0.71 | 0.43 | 0.68 | +0.28 | 52/135 | 0.35 (n=45) | 5/45 | 0.427 -> **0.452** |

## Diagnosis changes from the regrade (12)
| run | task | faults | diag old -> new | data_issues (start) |
|---|---|---|---|---|
| 62113495 | gym-010 | clean | 0.00 -> 1.00 | The labeled splits include both label and orientation_deg where label is exactly orientation_deg/45; this duplicates target information from |
| 62113495 | gym-031 | clean | 0.00 -> 1.00 | Pseudo-population construction mixes trials across 14 sessions (not simultaneous recordings), so neuron-neuron correlations are artificial/a |
| 62113495 | gym-032 | clean | 0.00 -> 1.00 | Pseudo-population construction mixes neurons across different sessions/patients so each trial is synthetic (not simultaneously recorded); th |
| 62113495 | gym-040 | clean | 0.00 -> 1.00 | Pseudo-population design mixes neurons across 20 sessions (not simultaneously recorded), so trial-wise correlations are artificial; also dat |
| 62113495 | gym-044 | clean | 0.00 -> 1.00 | Pseudo-population construction mixes neurons across different sessions/patients, so each trial is synthetic rather than simultaneously recor |
| ab0a4055 | gym-031 | clean | 0.00 -> 1.00 | Label leakage risk: in labeled splits, category is in one-to-one correspondence with y (y derived from category). category should not be use |
| ab0a4055 | gym-044 | clean | 0.00 -> 1.00 | Pseudo-population construction mixes neurons across 14 sessions (non-simultaneous units), so trial-wise noise correlations are artificial/ab |
| 280607cb | gym-008 | leak | 1.00 -> 0.00 | Only one 1000 ms time bin per trial, losing temporal structure; label duplicates category field on labeled splits (target leakage risk if ca |
| 280607cb | gym-020 | clean | 0.00 -> 1.00 | orientation_deg duplicates label information in labeled splits (perfect one-to-one mapping), which is target leakage if used as a feature; i |
| 280607cb | gym-032 | clean | 0.00 -> 1.00 | Pseudo-population construction breaks true trial simultaneity across neurons by combining different sessions; this can inflate/alter apparen |
| 280607cb | gym-035 | clean | 0.00 -> 1.00 | Pseudo-population construction mixes neurons across different sessions/patients, so rows are synthetic and ignore true trial-by-trial correl |
| 280607cb | gym-040 | clean | 0.00 -> 1.00 | Pseudo-population trials combine different sessions (not simultaneously recorded neurons), so cross-neuron noise correlations are lost and s |

## Notes
- Cost: 135 tasks, $5.26 total (metrics_total.json), ~$0.039/task. Verifier exit 0 on all 135; 1 run produced no predictions file.
- The 12 diag changes: 11 clean envs no longer false alarms (3 called the documented stimulus-log column "leakage";
  8 noted that the documented pseudo-population pools sessions, now ignored since mix is never planted in v4) and
  1 leak env loses a credit given for "label duplicates category field", not for the copied trials.
- Tasks whose recovered text does not reproduce the hosted diag keep the hosted diag (7 incl. the invalid one).
- Reproduce: cd ~/work/vals-hyp/broken/gym; python3 regrade_v4.py (needs full/outputs trajectories, full/<run>.json).
  Per-task rows: full/regraded_v4.json.
