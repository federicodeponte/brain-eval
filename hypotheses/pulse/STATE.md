# PULSE: heart rate from face-video colour numbers

Status: GO (data + validity done; no model has been run yet)

## Source and license
- UBFC-rPPG (first choice) is BLOCKED: public Google Drive folder, but every file (even 1 kB ground-truth txt) returns
  "Quota exceeded" on 2026-11-14. Dropped rather than routed through a personal Drive copy.
- Used instead: MCD-rPPG (Egorov et al., ACM MM 2025, "Gaze into the Heart"), https://huggingface.co/datasets/kyegorov/mcd_rppg,
  license CC BY 4.0. 600 subjects x {before, after 15 squats} x 3 cameras, 3-min videos, PPG synced to video frames.
- Subset used: frontal "FullHDwebcam" videos (640x480, 29.9 fps) of 75 subjects (40 before+after, 35 after-only) = 115 videos.
  Videos were streamed, reduced to face-ROI mean RGB per frame, and deleted. Only traces kept: ~/work/vals-hyp/_data/mcd/rgb2 (npz).

## Construction (extract.py, build.py)
- Face ROI: OpenCV Haar face detector over the first 5 s, median box, inner 60% width x 80% height of the face box, held fixed
  (subjects are seated still). Mean R,G,B per frame.
- Windows: 20 s starting at 30/70/110 s. Ground truth HR from the synced finger PPG; window kept only if FFT-peak HR and
  peak-interval HR agree within 5 bpm (313 windows pass).
- Buckets at the data quartiles: A under 68, B 68-80, C 80-92, D over 92 bpm. Items within 2 bpm of an edge excluded.
- One window per subject; 10 per bucket = 40 items. RGB resampled to 10 Hz (200 lines "R G B", 2 decimals).
- `tasks.jsonl`: 40 items, zero-shot, max 4.4k chars. Answers 10/10/10/10.
- `tasks_fewshot.jsonl`: + 4 labeled references (1 per bucket, other subjects, with exact bpm), max 19.5k chars.

## Validity gate (on exactly these 40 items, using only the numbers in the prompt)
- Chance: 25%.
- Training-free classic rPPG on the 10 Hz prompt data, FFT peak in 0.7-3.5 Hz -> bucket:
  - green-channel FFT: 70% (MAE 11.6 bpm)
  - POS (Wang 2017): 70% (MAE 10.4 bpm)
  - CHROM (de Haan 2013): 75% (MAE 13.6 bpm)
- These algorithms use no training data, so zero-shot and few-shot models have the same information as the baseline.
- Verdict: GO.

## Caveats
- Mean-RGB input is a stand-in for video; a model never sees pixels. A video-input variant would need the raw clips (not stored;
  could re-download per item from HF).
- Bucket edges are only 12 bpm wide, so a model that roughly knows "rPPG = periodic green dip" still has to count cycles precisely.
- MCD-rPPG is mostly a Russian adult cohort; post-exercise HR recovers during the 3-min clip, so a window's HR is an average.
- n=40 -> about +/-15 points CI.

## Files
- extract.py, build.py, tasks.jsonl, tasks_fewshot.jsonl, baseline.json, examples/example_zeroshot.txt, examples/example_fewshot.txt
