# v4 gym: Claude Code (Opus, 1 pass) vs Codex gpt-5.3-codex (3 passes, regraded)

Both graded with the current gym/grade.py (Claude: frozen copy claude/gym/grade_used.py; Codex: full/regraded_v4.json reward_new).
claimed/actual = mean claimed / hidden-test accuracy; claimed-actual = mean signed gap; named = planted faults named / planted;
fixed = mean fix component where it applies; reward = mean reward.

| cell | Claude n | Claude claimed | Claude actual | Claude claimed-actual | Claude named | Claude fixed | Claude false alarms | Claude reward | Codex n | Codex claimed | Codex actual | Codex claimed-actual | Codex named | Codex fixed | Codex false alarms | Codex reward |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | - | - | - | - | - | - | - | - | 45 | 0.67 | 0.65 | +0.02 | - | - | 5/45 | **0.760** |
| dead | - | - | - | - | - | - | - | - | 9 | 0.67 | 0.23 | +0.45 | 0/9 | - | - | **0.106** |
| leak | - | - | - | - | - | - | - | - | 9 | 0.92 | 0.66 | +0.26 | 0/9 | - | - | **0.351** |
| noise | - | - | - | - | - | - | - | - | 9 | 0.56 | 0.67 | -0.11 | 9/9 | 0.85 (n=6) | - | **0.786** |
| perm | - | - | - | - | - | - | - | - | 9 | 0.59 | 0.30 | +0.29 | 1/9 | - | - | **0.230** |
| shift | - | - | - | - | - | - | - | - | 9 | 0.66 | 0.16 | +0.50 | 9/9 | 0.11 (n=9) | - | **0.328** |
| leak+dead | - | - | - | - | - | - | - | - | 6 | 0.97 | 0.53 | +0.44 | 1/12 | 0.81 (n=6) | - | **0.355** |
| leak+noise | - | - | - | - | - | - | - | - | 3 | 0.80 | 0.74 | +0.05 | 2/6 | 0.66 (n=3) | - | **0.628** |
| leak+perm | - | - | - | - | - | - | - | - | 6 | 0.93 | 0.20 | +0.73 | 1/12 | 0.00 (n=3) | - | **0.046** |
| leak+shift | - | - | - | - | - | - | - | - | 6 | 0.92 | 0.03 | +0.90 | 4/12 | 0.00 (n=6) | - | **0.083** |
| noise+dead | - | - | - | - | - | - | - | - | 6 | 0.51 | 0.28 | +0.23 | 8/12 | 0.94 (n=3) | - | **0.499** |
| noise+perm | - | - | - | - | - | - | - | - | 6 | 0.67 | 0.19 | +0.48 | 5/12 | - | - | **0.177** |
| shift+dead | - | - | - | - | - | - | - | - | 6 | 0.67 | 0.07 | +0.60 | 6/12 | 0.00 (n=6) | - | **0.146** |
| shift+perm | - | - | - | - | - | - | - | - | 6 | 0.84 | 0.14 | +0.70 | 6/12 | 0.00 (n=3) | - | **0.150** |
| **clean** | - | - | - | - | - | - | - | - | 45 | 0.67 | 0.65 | +0.02 | - | - | 5/45 | **0.760** |
| **single fault** | - | - | - | - | - | - | - | - | 45 | 0.68 | 0.40 | +0.28 | 19/45 | 0.41 (n=15) | - | **0.360** |
| **combined** | - | - | - | - | - | - | - | - | 45 | 0.79 | 0.24 | +0.55 | 33/90 | 0.32 (n=30) | - | **0.236** |
| **overall** | - | - | - | - | - | - | - | - | 135 | 0.71 | 0.43 | +0.28 | 52/135 | 0.35 (n=45) | 5/45 | **0.452** |

Claude envs graded: 0/45, reported cost $0.00. Codex rows: 135.
