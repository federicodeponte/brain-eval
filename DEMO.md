# Demo script: "Can a model read a brain?" (3 minutes)

Screen: federicodeponte.github.io/brain-eval (open before you start, scrolled to top).

## 0:00 Hook (20 s)
"These are 40 real human neurons, recorded with electrodes in the hippocampus and amygdala of epilepsy patients while they looked at photos, pooled across recording sessions. We gave Claude the raw spike counts and three examples. No training. Can it tell what the person was looking at?"

## 0:20 The room plays (40 s)
Scroll to "Read a human brain". Show one question. "Darker means more spikes. Here's food, cars, people, animals. Here's the new row. What was it?"
Take a show of hands for A/B/C/D, click the most popular answer, reveal.
"That's hard for us. Here's how the models did."

## 1:00 Result (40 s)
Scroll to Results.
- "Human neurons, 40 questions: Opus 68%. A classic neuroscience decoder with the same three examples: 65%. Chance: 25%."
- "Mouse visual cortex, 60 questions: 87% vs 87%."
- "Hard tier, one example per class: Opus 50% on human (decoder 45%), 60% on mouse with 8 clips (decoder 50%). Far from solved."
- "Opus matches the decoder, but misses different questions, so it is reading the neurons its own way."
- "Haiku stays far behind."

## 1:40 Is it really learning? (30 s)
Scroll to the checks.
- "We swapped the labels: car neurons labeled 'food'. If Claude used brain trivia, it would pick the real category. It followed the examples 12 of 20 times, and the real category only 3."
- "With zero examples it just answers 'A' every time. With one, it gets 7 of 10. It learns from the examples in the prompt."

## 2:10 Through a mouse's eyes (25 s)
Scroll to the mouse section, click question 8.
"Here the mouse watched a film. Four candidate seconds. The decoder gets it, Opus doesn't."
Click a frame to reveal.

## 2:35 Close (25 s)
"The eval is on Valkyrie: human and mouse sets, five controls, and a hard tier where the best frontier model still gets only half right. First benchmark we know of where a model reads raw single-neuron recordings in its prompt. Code and data are open."

## If asked
- Data: DANDI 000004 (Chandravadia 2020, CC-BY), Allen Brain Observatory (Siegle 2021).
- Guessing: answer letters are balanced, Opus spreads its picks evenly, 70% right on non-A answers.
- With Python, Opus writes its own nearest-centroid decoder and makes the same picks as the decoder; no gain.
- Pooled pseudo-trials: human rows combine neurons from several sessions; the label-swap test shows the model reads the numbers, not the category names.
