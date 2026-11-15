# hear-the-room: where was this impulse response recorded? (4-way)

Status: GO (text variant built; per-item WAVs saved for a later audio variant)

## Source + license
- MIT Acoustical Reverberation Scene Statistics Survey (Traer & McDermott 2016, PNAS), 270 RIRs at 32 kHz,
  mcdermottlab.mit.edu/Reverb/IR_Survey.html (IRMAudio/Audio.zip, 11.7 MB). Free research download, no explicit license file; cite the paper.

## Construction (build.py)
- Labels from the survey location names (independent of any acoustic measurement), keyword-mapped:
  small enclosed (bathroom, shower, bedroom, car, pantry, laundry...) 27 | medium indoor (classroom, office, kitchen, bar, restaurant, shop...) 112 |
  large indoor hall (atrium, auditorium, gym, supermarket, stations, ballroom, theater, library, food court...) 31 | outdoor 72. Unmatched names dropped.
- 40 test items, 10 per class, letters balanced. Prompt (1.0-1.6k chars): Schroeder energy-decay curve in dB every 20 ms from the direct sound
  (until the IR ends, max 2 s, floor -80 dB) + early energy envelope every 2 ms for 0-58 ms.
- audio/room-NNN.wav = full IR per item (meta.audio_path) for an audio-input variant.
- Rebuild: `../_tools/.v4/bin/python -I build.py`

## Validity (features: EDT + T20 broadband and 3 bands, DRR, C50; trained on the 202 non-test IRs)
| method | acc |
|---|---|
| chance | 0.25 |
| logistic regression, class-balanced | 0.65 |
| logistic regression, unbalanced | 0.55 |
| kNN-5 | 0.475 |
| broadband T20 only, nearest class median | 0.525 |

Confusion (balanced LR): outdoor 10/10, small 7/10, large 5/10, medium 4/10.

## Caveats
- MIT IRs are trimmed (<= 0.8 s) at the noise floor, so curve length is itself a cue (outdoor curves end at 120-240 ms). Physical, not a pipeline artifact.
- Labels are room type, not measured volume: some "medium" rooms are big, some "large" halls are damped, so the ceiling for any method is < 1.
- RT60-bucket variant not built: its label would come from the same Schroeder method as the baseline (circular).
- No memorization risk.
