# human-neurons: which photo category was a human looking at? (single-neuron spike counts)

Status: DATA + VALIDITY done for source 1 (DANDI 000004). GO. No LLM calls made.

## Source
- DANDI 000004, "A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task"
  (Chandravadia et al. 2020, Scientific Data; data from Faraut et al. 2018, Rutishauser lab, Cedars-Sinai / Huntington).
- License: CC-BY-4.0 (DANDI metadata). 87 NWB sessions, amygdala + hippocampus microwires.
- Ethics: de-identified, publicly released research data from epilepsy patients who consented to research recordings
  during clinical depth-electrode monitoring (IRB approved per the paper). We use only spike times and trial labels,
  no images, no subject info.
- Access: streamed with remfile/h5py from the DANDI API (only trials + units read). Local footprint: 1.6 MB of npz in data004/.
  No raw NWB on disk.

## Construction
- Task per session: 200 photo presentations (100 learn + 100 recognition), 5 categories, 1 s on screen.
- Spike count per unit in [200, 1200] ms after photo onset.
- Single sessions are too small (median 5-fold LR 4-way acc 0.33; only 1/87 sessions >= 0.60), so we use a
  pseudo-population: the 25 sessions sharing image set {cars, food, people, spatial, animals} (575 units), drop "spatial",
  keep 4 classes (cars, food, people, animals).
- Per session, trials split 50/50 (stratified, seed 0): selection half used ONLY to pick the 40 most category-selective
  units (one-way ANOVA p, mean count >= 0.5); item half used for every example/query row. No overlap.
- A row = pseudo-trial: for each contributing session, one item-half trial of that row's category; all of a session's
  units use the same trial. Within each session the 13 rows of an item use 13 distinct trials and distinct pictures
  (query picture never appears in that session's example rows). Asserted in code.
- Prompt: 4 classes x 3 labeled example rows + 1 query row, 40 neurons with region tags (LA/RA/LH/RH), class names shown,
  "Answer with a single letter." ~3.4k chars.
- 40 items, letters 10/10/10/10, query categories 10/10/10/10, letter x category crossed (2-3 each).

## Validity gate (tasks.jsonl, seed 11)
| measure | acc |
|---|---|
| chance | 0.25 |
| nearest centroid, correlation distance, sqrt counts, in-prompt examples only | 0.65 (per class: cars 5/10, food 9/10, people 8/10, animals 4/10) |
| logistic regression on held-out pseudo-trials (trained on 240 pseudo-trials excluding the query's trials/pictures) | 0.775 |
Robustness (earlier build, unbalanced query categories): seeds 11/12/13 -> NC 0.70/0.75/0.725, LR 0.90/0.825/0.775.
Sweep (scripts/sweep_pseudo.py): other image sets weaker (houses set NC 0.41-0.56, fruit set NC 0.40-0.59); K=5 helps (NC 0.77 for this set).

## Files
- tasks.jsonl (40 items), examples/human-mtl-category-000.txt and -001.txt, build_stats.json
- scripts/: common.py (DANDI streaming), extract004.py (per-session counts), survey004.py (single-session decodability),
  pseudo.py + sweep_pseudo.py (pseudo-population), build_pseudo.py (items + baselines). env: .venv (uv), source env.sh first.
- Rebuild: `. ./env.sh; SEED=11 .venv/bin/python -I scripts/build_pseudo.py $PWD/scripts tasks.jsonl`

## Caveats
- Pseudo-population: units come from different patients/sessions, so trial-to-trial noise correlations are destroyed;
  this is standard in decoding papers but is not "one brain, one moment". The prompt says so.
- Unit selection uses category selectivity from the disjoint selection half; that is a fair preprocessing step but
  makes the population more informative than a random 40 units.
- Class names are shown; a model could use priors (e.g., "people/faces drive MTL most"), but rows carry no category info
  beyond spikes and examples, and letters/categories are balanced.
- Both learn (first viewing) and recognition (some repeated) presentations are in the pool; repeats of the same picture
  are kept out of an item within a session.
- Response window fixed at 200-1200 ms; rates for some units are low (0-3 spikes).

## Other sources checked
- DANDI 000623 (Keles et al. 2024, movie "Bang! You're Dead", CC-BY-4.0, 29 sessions ~0.5-1.3 GB each, units streamable):
  FIX, not built. The movie is watched once, so there are no independent repeats per segment; few-shot "which segment"
  from adjacent time bins would be decodable from slow drift, not content. Scene-cut/face annotations are not in the NWB.
  Recognition phase has only 41 clips/session. Viable if annotations from the paper's repo are joined (face on screen vs not).
- DishBrain (Cortical Labs, Kagan et al. 2022 Neuron): DROP. No public raw spike dataset found; only a visualiser
  (spikestream.corticallabs.com) and visualisation code on GitHub.
- OpenNeuro iEEG / DANDI 000469: not needed; 000469 (Sternberg WM, CC-BY-4.0, 9.8 GB) is a backup with the same lab format.
