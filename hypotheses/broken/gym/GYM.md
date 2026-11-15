# Broken-instrument decoding gym

An RL gym where an agent gets real spike recordings, builds a decoder, predicts held-out trials, says how
accurate it will be, and reports what is wrong with the data. Most environments carry one or two planted
instrument faults. Each fault can be found by a careful analyst from the visible files. Every environment is
generated from public recordings: Allen Visual Coding Neuropixels session 732592105 (natural movies, natural
scenes, drifting and static gratings; VISp/VISal/VISpm/VISrl) and DANDI 000004 (human MTL single neurons).

## Generate environments
```
P=~/work/vals-mouse/.venv/bin/python; cd ~/work/vals-hyp/broken/gym
$P gen.py --seed 17 --out envs/                  # one env, faults drawn from the seed
$P gen.py --seed 17 --faults leak,dead --out envs/   # force the faults (clean = no fault)
$P gen.py --validate --seeds 0:500               # naive / fault-aware / clean-reference accuracy per env (jsonl)
$P build_gym.py OUT 15 3 15                      # a balanced Harbor dataset: 15 clean, 3 per single fault, 15 combined
```
The same seed always produces the same env. Each seed randomizes the recording, stimulus subset, area, unit subset
and order, split sizes, time bins, faults and their strength, file names, array keys and axis order.
Faults: `leak` (val near-copies of train, duplicated train rows), `shift` (labels lag the stimulus log),
`noise` (25-35% of labels shuffled), `mix` (experimental, not in v4: 35-50% of rows from a different unit population),
`dead` (top units silent in test only), `perm` (test column order permuted), plus pairs of these.

## Reward (tests/grade.py, deterministic, stdlib only)
- skill = (acc - chance) / (1 - chance), clipped to [0, 1]
- calib = 1 - 2 |claimed_accuracy - acc|, clipped to [0, 1]
- diag: clean env = 1 if nothing is flagged. Faulted env = fraction of planted faults named, minus 0.25 per
  unrelated fault group flagged (keyword families with negation handling).
- fix (only if the fault hurts a naive decoder and can be undone from the visible files) = 1 if the hidden-test
  accuracy is within 0.08 of the clean twin's reference accuracy, falling linearly to 0 at 0.28 below it.
- reward = mean of the components that apply. All components are in `details.json`.

## Why it is verifiable
The generator holds the true labels of the test trials and the exact fault it planted, and it computes the
clean-twin reference accuracy by fitting the same reference decoder to the unfaulted copy. Grading is plain
arithmetic on the hidden labels plus a keyword match. No LLM judge, no human labels. The number of environments
is limited only by the seed space: recordings x stimulus subsets x units x splits x faults.
