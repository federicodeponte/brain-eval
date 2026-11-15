# HEARTBEAT: rhythm from raw ECG numbers

Status: GO (data + validity done; no model has been run yet)

## Source and license
- PhysioNet/Computing in Cardiology Challenge 2017, "AF Classification from a Short Single Lead ECG Recording" v1.0.0
  (https://physionet.org/content/challenge-2017/1.0.0/), training2017.zip (8,528 AliveCor recordings, 300 Hz, labels N/A/O/~).
- License: Open Data Commons Attribution License v1.0 (ODC-By). Cite Clifford et al., CinC 2017.
- Local copy: ~/work/vals-hyp/_data/cinc/training2017 (211 MB).

## Construction (build.py)
- 4 classes = the challenge labels: A Normal sinus rhythm (N), B Atrial fibrillation (A), C Other rhythm (O), D Too noisy (~).
- Records of 12 s or longer only; 10 items per class sampled at random (seed 7); 2 more per class held out as few-shot examples.
- Window: seconds 2-12 of each record (skips the start-up transient). Resampled 300 Hz -> 100 Hz (zero-shot) or 50 Hz (few-shot), rounded to integer microvolts.
- `tasks.jsonl`: 40 items, zero-shot, 100 Hz, prompt max 4.7k chars. Answers A/B/C/D = 10/10/10/10.
- `tasks_fewshot.jsonl`: same 40 items at 50 Hz, plus 8 labeled reference recordings (2 per class) in each prompt, max 18.4k chars.
- Item order shuffled; answer letter = fixed class order (labels always listed A-D in the same order).

## Validity gate (on exactly these 40 items)
- Chance: 25%.
- Supervised baseline: 14 hand features (R-peak count, RR mean/SD/CV, RMSSD, fraction of |dRR|>50 ms, amplitude, high-/low-frequency power share),
  trained on 3,000 other CinC records (no item or few-shot record in training):
  - 100 Hz: logistic regression 57.5%, random forest 67.5%
  - 50 Hz:  logistic regression 77.5%, random forest 67.5%
- Prompt-only baseline (few-shot file): 1-nearest-neighbour on the same features using only the 8 examples in the prompt: 45-50%.
- Verdict: clearly above chance -> GO.

## Caveats
- n=40, so each accuracy has about +/-15 points of 95% CI; the LR/RF spread (57-78%) is noise at this n.
- "Other rhythm" is a heterogeneous bucket (PVCs, flutter, bradycardia...), and the AF-vs-Other boundary is hard even for the feature baseline (AF recall 30% in RF); a 3-way N/AF/noisy version would be cleaner if the model scores look like noise.
- The 2017 challenge data is widely known; a model could know the clinical criteria (irregularly irregular RR, absent P waves) but has never seen these exact number sequences as text.

## Files
- build.py, tasks.jsonl, tasks_fewshot.jsonl, baseline.json, examples/example_zeroshot.txt, examples/example_fewshot.txt
