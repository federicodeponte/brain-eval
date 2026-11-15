# Plan to demo (Sun Nov 15, 2pm PT)

Story: **"Can a model read a brain?"** Frontier models get raw neuron spike counts + 3 labeled examples, no training. We score them against a classic decoder and prove with controls that they really learn.

| # | Track | Owner | Status | Done by |
|---|---|---|---|---|
| 1 | **Movie frames**: mouse options shown as the real movie stills; spikes + 4 frames + Opus pick + decoder pick | Claude | Extracting frames from Allen data | Sat 11pm |
| 2 | **Room plays along**: "Try one yourself" with 10 human-neuron questions, live leaderboard humans vs Haiku vs Opus vs decoder | Claude builds, Federico runs it at the demo | Building | Sat midnight |
| 3 | **Code twist**: Opus with Python on the same 20 questions | Claude | Running. Early: Opus writes a nearest-centroid decoder and matches its picks 9/9 | Sat 10pm |
| 3b | **Full Opus run** on every question (human 40, mouse 60 + 30) so all numbers are same-question comparisons | Claude | Queued after 3 | Sun 1am |
| 4 | **Vals run**: GPT, Gemini, open Qwen; all sets incl. controls; confidence intervals | Federico: get depontefede@gmail.com invited to the Valkyrie org (ask Olivia, Langston or Jarett). Claude: push + run | Blocked on invite | Sun 10am |
| 5 | **Demo**: final page, 3-minute script, one headline chart | Claude drafts, Federico rehearses | After 3b | Sun 11am |
| 6 | **After**: arXiv preprint; outreach to Rutishauser lab + Allen Institute; @ValsAI post | Claude drafts, Federico approves every message | - | Next week |

## Headline (decided by data, not before)
- If Opus >= decoder on the full sets: "Claude reads a human brain from 3 examples, as well as a classic decoder, and the controls prove it is learning."
- If Opus < decoder: "Frontier models can't yet read a brain in their head, but they build the tool that does."
- Hard tier for the track ("tasks frontier models still fail"): 8 clips, fewer neurons, shorter windows; Haiku already fails.

## Rules
- Every number on the page is same-question, with n shown. No claim before the full run.
- Subscription runs only within Federico's approved count and cap (now $125, 300 jobs).
- No outbound message without Federico's OK on the exact text.
