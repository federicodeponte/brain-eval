# Demo script: "Can a model read a brain?" (3 minutes)

Screen: federicodeponte.github.io/brain-eval (open before you start, scrolled to top).

## 0:00 Hook (20 s)
"We gave AI agents real neuron recordings, from mice and from epilepsy patients, and asked them to build a decoder and tell us how accurate it is. Some copies of the data were broken, the way real lab data breaks. The question: does the agent notice, or does it report a number it can't back up?"

## 0:20 The hard part (50 s)
Scroll to "The hard part".
- "Plain code grades it: real accuracy on a hidden test, the gap to what the agent claimed, and whether it named the fault. No model judge."
- "Codex, on broken data: claims 83%, gets 42%. Leaked trials: it claims 98%. It never names that fault."
- "Shifted clock: Codex names the fault every time, and still reports the inflated number."
- "Claude Opus on the same tasks: claims 72%, gets 63%, names 11 of 12 faults. The benchmark tells agents apart."

## 1:10 The gym (35 s)
Scroll to "The gym".
- "One seed, one new environment: random recording, split, faults, strength. 45 so far, five fault types, alone or combined."
- "Two faults at once: Codex claims 79% and gets 24%. Opus gets 61%."
- "But leaked trials fool Opus 8 times out of 10. That's the open problem, and the gym can make endless versions of it to train on."

## 1:45 Can it read neurons at all? (35 s)
Scroll to Results.
- "Before any of that: can a model read spike counts at all? Multiple choice, three examples per class in the prompt, no training."
- "Human neurons: Opus 68%, a classic decoder 65%, GPT-5 55%, chance 25%. Mouse: Opus 87%, decoder 87%."
- "Hard tier, one example per class: Opus 50% and 60%. Far from solved."

## 2:20 Is it really reading? (25 s)
Scroll to the checks.
- "Swap the labels and it follows the examples, 12 of 20, not brain trivia."
- "Zero examples: it answers A every time. One example: 7 of 10."
- "Relabel the neurons as machine sensors: same score. It reads the numbers in the prompt."

## 2:45 Close (15 s)
"It's all on Valkyrie: the reading tier, the broken-instrument tier, and the gym generator. Real recordings, graded by code, and the best agent still misses most leaked trials. Code and data are open."

## If asked
- Data: DANDI 000004 (Chandravadia 2020, CC-BY), Allen Brain Observatory (Siegle 2021).
- Grader bug: an earlier grader counted the agent calling a documented label column "leakage" as naming the leak fault. Fixed and noted in `broken/regraded.md`; all numbers above use the fixed grader.
- Runs: Codex 3 passes on Valkyrie; Opus 1 pass via Claude Code in an isolated local sandbox, same tasks and grader.
- Faults are validated: a naive pipeline shows a 30-73 point gap on each fault while the hidden test stays honest, and each fault leaves a trace a careful analyst can find.
- Pooled pseudo-trials: human rows combine neurons from several sessions; the label-swap test shows the model reads the numbers, not the category names.
