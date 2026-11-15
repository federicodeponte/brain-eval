# SLEEP: sleep stage from 30 s of raw EEG numbers

Status: GO (data + validity done; no model has been run yet)

## Source and license
- Sleep-EDF Database Expanded (PhysioNet, https://physionet.org/content/sleep-edfx/1.0.0/), Sleep Cassette study, healthy adults,
  fetched with `mne.datasets.sleep_physionet.age.fetch_data(subjects=0..11, recording=[1])` = 12 subjects, night 1 each.
- License: Open Data Commons Attribution License v1.0 (ODC-By). Cite Kemp et al. 2000 and Goldberger et al. 2000 (PhysioNet).
- Local copy: ~/work/vals-hyp/_data/mne/physionet-sleep-data (262 MB).

## Construction (build.py)
- Channel EEG Fpz-Cz, 100 Hz -> resampled to 50 Hz, integer microvolts, one 30 s scoring epoch = 1,500 numbers.
- Labels from the expert hypnogram (R&K): A Wake, B Light (S1+S2), C Deep (S3+S4 = N3), D REM. Movement/unscored epochs dropped;
  Wake limited to within 30 min of the sleep period (removes daytime/in-bed-awake recording).
- Item subjects: SC400-SC403 (4 people). 10 epochs per class sampled at random from them (seed 11).
- `tasks.jsonl`: 40 items, zero-shot, max 5.3k chars. Answers 10/10/10/10.
- `tasks_fewshot.jsonl`: same 40 items + 4 labeled reference epochs (1 per class, from training subjects), max 23.4k chars.
  Only 1 shot per class fits under ~30k chars at 50 Hz.

## Validity gate (on exactly these 40 items)
- Chance: 25%.
- Supervised baseline: log band powers (0.5-2, 2-4, 4-8, 8-12, 12-16, 16-25 Hz) + relative powers + amplitude, logistic regression
  trained on the other 8 subjects (no item subject in training): 82.5% (per class: Wake 70%, Light 60%, Deep 100%, REM 100%).
- Prompt-only baseline: 1-NN on the same features using only the 4 prompt examples: 35%.
- Verdict: GO.

## Caveats
- Single-channel EEG without EOG/EMG; REM vs N1 is normally separated using eye movements and chin tone, so ceiling is below
  full-PSG scoring (human inter-rater agreement on 5 stages is ~80%).
- Few-shot is thin (1 example per class) due to the character budget; the 1-NN-on-prompt baseline is weak (35%), so the few-shot
  variant tests "can you learn from one example", not "same info as a strong baseline".
- n=40 -> about +/-15 points CI.

## Files
- build.py, tasks.jsonl, tasks_fewshot.jsonl, baseline.json, examples/example_zeroshot.txt, examples/example_fewshot.txt
