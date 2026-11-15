# Mind-reading a mouse: dataset state (2026-11-14)

## Status
v1 dataset built and validated. No LLM calls made. Next step: upload tasks.jsonl to Vals and run the models.

## Data
- Allen Brain Observatory Visual Coding Neuropixels, public S3 (no credentials needed), session 732592105 (brain_observatory_1.1)
- data/session_732592105.nwb (2.9 GB) plus the ecephys-cache CSVs (units, channels, probes, sessions, metrics) in data/
- 111 VISp units pass the default quality filters (isi_violations<0.5, amplitude_cutoff<0.1, presence_ratio>0.95)
- Stimulus: natural_movie_three (3600 frames at 30 Hz = 120 s), 10 complete repeats

## Item construction (build.py)
- The movie is cut into 120 non-overlapping 1 s segments; a clip is one segment. Onsets come from the presentation time of each segment's first frame.
- Features: spike counts in 10 bins of 100 ms for the top 40 units, so each trial is a 40x10 integer matrix. Units are ranked by split-half reliability (odd vs even repeats, correlation of the full-movie PSTH). The top 40 range from r=0.62 to 0.85.
- Each item has K candidate clips (random segments) labeled A..; 3 example trials per clip come from random repeats other than the query repeat; 1 query trial comes from the held-out repeat. Answers are balanced and query repeats are spread across items. An assert checks that the query repeat never appears among the examples.
- 60 items with K=4 and 30 with K=8 (seed 0)

## Prompts
Plain text: 2-3 sentences of instruction, then the matrices (rows = neurons, columns = bins, space-separated), ending with "Answer with a single letter."
K=4 is about 11.4k chars; K=8 is about 21.3k chars.

## Baselines (each uses only the item's own few-shot examples)
| set | chance | nearest centroid (corr) | 1-NN (corr) | logreg (log1p, z-scored) |
|---|---|---|---|---|
| K=4, 60 items | 0.25 | 0.867 | 0.817 | 0.933 |
| K=8, 30 items | 0.125 | 0.667 | 0.800 | 0.767 |
| K=4, 1000 items (robust/) | 0.25 | 0.835 | 0.789 | 0.923 |
| K=8, 1000 items (robust/) | 0.125 | 0.715 | 0.661 | 0.893 |

## Files
- tasks.jsonl: 90 items with id, k, prompt, answer, session_id, unit_ids, segment_ids, query_repeat, plus extras (example_repeats, stimulus, bin_ms, n_bins)
- examples/k4_example.txt, examples/k8_example.txt: prompts for human inspection, with [ANSWER: X] appended (not part of the task)
- build_report.json: config, baselines, prompt sizes, unit reliabilities; robust/build_report.json: the 1000-item check
- build.py: rebuild with `.venv/bin/python build.py [--units 40 --bins 10 --binms 100 --n4 60 --n8 30 --seed 0]`
- pick.py: ranks sessions by good VISp units
- Env: .venv (python 3.11, numpy, pandas, h5py, scikit-learn). Set UV_CACHE_DIR=.uvcache because ~/.cache is not writable for this user.

## Caveats
- Units are selected using reliability across all 10 repeats. This is dataset-level selection, not per-item leakage, but the selection does see the query repeat. Switch to a training-repeat-only selection for a stricter version.
- Some rows are all zeros in a given trial (low-rate units).
- /home on AX41 is at 96% (about 8.5 GB free). Delete data/*.nwb after the final build if space is needed.

## v2 (2026-11-14)
- tasks_v2.jsonl (60 K=4 + 30 K=8): units chosen per item from non-query repeats only (fixes the caveat above). NC 0.817/0.833, LR 0.967/0.867. Built by ~/work/vals-hyp/controls/scripts/mouse_controls.py; report build_report_v2.json; Valkyrie folder valkyrie/dataset-v2. Controls: ~/work/vals-hyp/controls/STATE.md
