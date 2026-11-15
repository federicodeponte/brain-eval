# Can a model read a brain?

Vals AI Hackathon (Nov 14-15 2026), Build Your Own Eval track, built on [Valkyrie](https://github.com/vals-ai/Valkyrie).

Frontier models get raw spike counts from real neurons plus a few labeled examples in the prompt, and must decode what the animal or person was seeing. A 10-line decoder (nearest centroid on the same in-prompt examples) is the baseline.

## Main tasks
| Task | Data | Chance | Decoder | Haiku | Opus (pilot, n=10) |
|---|---|---|---|---|---|
| Human neurons: car / food / person / animal photo | DANDI 000004, 40 amygdala/hippocampus neurons, epilepsy patients | 25% | 4/10 on the same items (65% over all 40) | 5/10 | 5/10 |
| Mouse: which of 4 movie clips | Allen Neuropixels 732592105, 40 VISp units x 10 bins | 25% | 7-9/10 on the same items | 6/10 | 8/10 |
| Mouse: which of 8 movie clips | same | 12.5% | 5-6/10 on the same items | 0/10 | 7/10 |

## Controls (does the model learn from the examples?)
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

## Control results (Opus, pilot)
- **Label-swap (human, n=20):** followed the in-prompt example labels 12/20, the true category 3/20 (decoder 65% / 12.5%). The model reads the neurons, not brain trivia.
- **Shot-curve (mouse K=4):** 0 examples: answered "A" every time (chance; 5/10 only because 5 of those 10 answers were A). 1 example: 7/10.

## Data credits
- Chandravadia N. et al. (2020) A NWB-based dataset and processing pipeline of human single-neuron activity during a declarative memory task. *Scientific Data* 7:78. DANDI 000004, CC-BY-4.0. Patients with drug-resistant epilepsy implanted with depth electrodes for seizure localization.
- Faraut M.C.M. et al. (2018) Dataset of human medial temporal lobe single neuron activity during declarative memory encoding and recognition. *Scientific Data* 5:180010. DANDI 000004, CC-BY-4.0.
- Siegle J.H. et al. (2021) Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature* 592:86-92. Allen Brain Observatory Neuropixels, Allen Institute terms of use.

## Prior work check (Nov 14 2026)
No published work found that gives a frontier LLM raw single-neuron spike counts + labeled examples in-context (no tools, no training) and scores it against a decoder. Nearest: LLM agents that write analysis code or call trained decoders (NS-Copilot EMNLP 2026, SpikeAgent 2025, SpikeLab 2026, DANDI LLM-notebook tools 2025, Neurodata Without Boredom 2026); trained brain encoders feeding LLMs (BrainLLM, Neurrator, Thought2Text).

**Correction (Nov 14):** earlier decoder numbers were full-set averages; on the exact pilot items Opus is level with the decoder. n=10 per cell is too small to rank; full run pending.
