# Can a model read a brain?

Vals AI Hackathon (Nov 14-15 2026), Build Your Own Eval track, built on [Valkyrie](https://github.com/vals-ai/Valkyrie).

Frontier models get raw spike counts from real neurons plus a few labeled examples in the prompt, and must decode what the animal or person was seeing. A 10-line decoder (nearest centroid on the same in-prompt examples) is the baseline.

## Headline: agents claim 83%, get 42%
**Broken-instrument tier** (`broken/`, Valkyrie dataset `brain/broken@v3`): an agent gets real spike recordings, builds a decoder, predicts held-out trials and reports its own accuracy. Each of 12 recordings (8 mouse, 4 human) comes as a clean copy and a copy with one planted fault. Scored by plain code, no model judge: real accuracy on a hidden test, gap between claimed and real accuracy, and whether the fault was named.

Codex (gpt-5.3-codex), 3 passes x 24 tasks:

| Data | Claimed | Real (hidden test) | Fault named |
|---|---|---|---|
| Clean | 75% | 71% | false alarm 6/36 |
| Leaked trials | 98% | 65% | 0/12 |
| Shifted clock | 80% | 24% | 12/12, never fixed |
| Dead channels | 71% | 36% | 8/12 |
| **All faulty** | **83%** | **42%** | |

Mean reward 0.79 on clean data, 0.39 on faulty data (scored with the fixed grader, `broken/regraded.md`: an earlier grader version counted the agent calling a documented label column "leakage" as naming the leak fault). A naive pipeline shows the same gaps (validation in `broken/STATE.md`): each fault inflates the in-sample estimate by 30-73 points while the hidden test stays honest.

### Gym: endless environments, harder faults (`gym/`, Valkyrie `brain/broken-gym`)
`gym/gen.py --seed N` builds a fresh environment from real recordings (random recording, units, split, faults and fault strength; same seed = same files). 5 validated fault types: leaked trials, shifted clock, dead channels, mislabelled trials, channels shuffled between train and test files; single or combined. Reward adds credit for actually fixing a fault (test accuracy back to the clean level). 45 environments so far: 15 clean, 15 single-fault, 15 combined.

Codex (gpt-5.3-codex), 3 passes x 45 (`gym/full_results.md`):

| Data | Claimed | Real | Reward |
|---|---|---|---|
| Clean | 67% | 65% | 0.76 |
| One fault | 68% | 40% | 0.36 |
| Two faults | 79% | 24% | 0.24 |

It names the shifted clock 9/9 and still trains on the bad labels (fixed 0.11); leaked trials, dead channels and shuffled channels go unnamed in 26 of 27 runs.

## Reading tier: multiple choice, every question, same items for every solver
Opus via claude -p; GPT-5 on Valkyrie (oneshot-mcq agent, Modal sandboxes).

| Task | n | Opus | GPT-5 | Decoder | Chance | only Opus right / only decoder right |
|---|---|---|---|---|---|---|
| Human neurons | 40 | 68% | 55% | 65% | 25% | 5 / 4 |
| Mouse, 4 clips | 60 | 87% | 75% | 87% | 25% | 5 / 5 |
| Mouse, 8 clips | 30 | 67% | 47% | 67% | 12.5% | 3 / 3 |
| Human, hard (1 example per class) | 20 | 50% | 40% | 45% | 25% | 5 / 4 |
| Mouse, hard (8 clips, 1 example each) | 20 | 60% | 55% | 50% | 12.5% | 3 / 1 |

Opus ties a nearest-centroid decoder on the three main tasks from 3 in-prompt examples per class, but solves different items. On the hard tier (1 example per class, built in hard/) Opus stays level with the decoder at 50% and 60%: far from solved. Decoders trained on many more trials do much better (73% mouse, 97% human), so the signal is there, but they see far more data than the one example in the prompt. Answer letters are balanced; Opus picks are spread evenly across letters. GPT-5 used its whole 16,000-token thinking budget without answering on 4 human and 7 mouse questions (counted wrong); on mouse hard it ran with a 48,000 budget and answered all.

## Controls (does the model learn from the examples?)
**Result, no-brain-words control:** the same 20 human questions relabeled as pulse counts from 40 machine sensors (modes A-D, no neuron/photo/region words). Opus 13/20 both ways (1 item right only as sensors, 1 only as neurons). The model reads number patterns from the in-prompt examples, not neuroscience priors: this benchmark measures in-context reading of real neural data.

- **Label-swap (human, n=20):** followed the in-prompt example labels 12/20, the true category 3/20 (decoder 65% / 12.5%). The model reads the neurons, not brain trivia.
- **Shot-curve (mouse K=4):** 0 examples: answered "A" every time (chance; 5/10 only because 5 of those 10 answers were A). 1 example: 7/10.

In `hypotheses/controls/` and `mouse/valkyrie/dataset-*`:
- **swap-out**: true clip examples replaced by a clip from another stimulus; correct = E) none of these
- **label-swap**: human category names permuted; does the model follow the examples or its priors?
- **shot-curve**: 0/1/3/5 examples per option; learning curve
- **null**: shuffled / Poisson data, expect chance
- **planted**: obvious planted pattern, expect ~100% (harness check)

## Layout
- `mouse/` build scripts, tasks (`tasks_v2.jsonl` = units chosen from training repeats only), `valkyrie/` datasets, one-shot MCQ agent, grader, leaderboard
- `hypotheses/human-neurons/` human set; `controls/` learning controls; `pilot/` claude -p pilot runner + results
- `hypotheses/*` other explored hypotheses (taste, yield, protein, connectome, room, whale, heartbeat, sleep, pulse) with STATE.md each

## Run on Valkyrie
See `mouse/valkyrie/STATE.md`: `valkyrie dataset push ...`, `valkyrie agent push mouse/valkyrie/agents/oneshot-mcq`, `valkyrie run start ...`, `python3 leaderboard.py results-*.json --md`.

Raw recordings are not included (download scripts in each folder). Data licenses: Allen Institute terms, DANDI CC-BY-4.0, PhysioNet ODC-By.

## Data credits
- Chandravadia N. et al. (2020) A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task. *Scientific Data* 7:78. DANDI 000004, CC-BY-4.0. Patients with drug-resistant epilepsy implanted with depth electrodes for seizure localization.
- Faraut M.C.M. et al. (2018) Dataset of human medial temporal lobe single neuron activity during declarative memory encoding and recognition. *Scientific Data* 5:180010. DANDI 000004, CC-BY-4.0.
- Siegle J.H. et al. (2021) Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature* 592:86-92. Allen Brain Observatory Neuropixels, Allen Institute terms of use.

## Prior work check (Nov 14 2026)
No published work found that gives a frontier LLM raw single-neuron spike counts + labeled examples in-context (no tools, no training) and scores it against a decoder. Nearest: LLM agents that write analysis code or call trained decoders (NS-Copilot EMNLP 2026, SpikeAgent 2025, SpikeLab 2026, DANDI LLM-notebook tools 2025, Neurodata Without Boredom 2026); trained brain encoders feeding LLMs (BrainLLM, Neurrator, Thought2Text).

