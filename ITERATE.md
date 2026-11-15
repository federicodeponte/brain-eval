# Iterate AI Research Hackathon: AI Scientist Swarms track

Team Brain Eval. Same repo as the Vals submission; this file is the research-process view.

## One line
A swarm of research agents searched for a benchmark that frontier models struggle on, killed the ideas that didn't hold up, and caught the agents overclaiming, which became a gym for training honest AI scientists.

## What the swarm did
1. **Proposed 12 directions** for "can a model read raw biological signals from a few examples": mouse visual cortex, human neurons, taste, reaction yield, protein mutations, fly connectome, room acoustics, whale codas, ECG rhythm, sleep EEG, pulse from face video, and broken-instrument decoding. One agent per direction, each with its own `hypotheses/<name>/STATE.md`.
2. **Gated every idea on a non-LLM baseline.** An idea survives only if a simple decoder beats chance on its questions, so the benchmark measures reading, not noise.
3. **Ran controls before trusting any score:** label swap, 0/1-shot learning curve, swapped clip, null data, and a no-brain-words relabel. The relabel showed the models read number patterns, not neuroscience knowledge, so the claim was narrowed to match.
4. **Hit a wall, then turned it into the result.** On the reading tier Opus ties a classic decoder, so it's not hard enough. The next step was giving agents broken data to analyze. Codex claimed 83% and got 42%.
5. **Built the gym.** `gym/gen.py --seed N` makes a fresh environment from real recordings with 5 validated fault types, alone or combined. Grading is plain code: real accuracy on a hidden test, the gap to the claim, whether the fault was named, and whether it was fixed.
6. **Caught its own bug.** The first grader counted the phrase "metadata leakage" as naming the leak fault. We found it, fixed it, re-scored, and logged it (`broken/regraded.md`).

## Results (same tasks, same grader)
| Agent | Two-fault data: claimed | Real | Gym reward |
|---|---|---|---|
| Codex (gpt-5.3-codex), 3 passes | 79% | 24% | 0.45 |
| Claude Opus (Claude Code), 1 pass | 71% | 61% | 0.76 |

Open problem: leaked (duplicated) trials fool Opus 8 times out of 10. The gym can generate endless environments with that fault, for training and for testing.

## Why it matters for AI scientists
An AI scientist that reports numbers it can't back up is worse than none. This gym measures that failure on real lab data, and it can generate new environments to train against it.

## 2-minute pitch
- 0:00 "We let a swarm of research agents look for a benchmark that frontier models fail. They proposed 12 ideas, from neurons to whale clicks to sleep EEG, and only kept the ones a simple decoder could solve."
- 0:25 "The brain one worked: models read raw spike counts as well as a classic decoder. So reading wasn't the hard part."
- 0:45 "Analysis was. We gave agents broken recordings and asked for their accuracy. Codex claimed 83% and got 42%."
- 1:05 "So we made it a gym: one seed, one new broken environment, graded by code. With two faults, Codex claims 79% and gets 24%. Opus gets 61%, but leaked trials still fool it 8 times out of 10."
- 1:30 "We even caught our own grader overcounting, and fixed it. It's all in the repo, including the dead ends."
- 1:50 "Open source, on Valkyrie. An AI scientist should know when its data is broken."

Links: github.com/federicodeponte/brain-eval · federicodeponte.github.io/brain-eval
