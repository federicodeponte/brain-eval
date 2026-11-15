# Decode what the brain was seeing

You have spike-count recordings from 27 single units in mouse anterolateral visual area (VISal), Allen Brain Observatory Neuropixels, while the mouse watched natural movie one (30 s clip), replayed 20 times.

Each trial is a 10 x 27 matrix: 10 time bins of 50 ms by 27 units starting at stimulus onset. The task is to decode which of 10 stimulus classes was shown on each trial:

- 0: movie time 10-10.5 s (0.5-second segment starting at 10 s)
- 1: movie time 10.5-11 s (0.5-second segment starting at 10.5 s)
- 2: movie time 11-11.5 s (0.5-second segment starting at 11 s)
- 3: movie time 11.5-12 s (0.5-second segment starting at 11.5 s)
- 4: movie time 12-12.5 s (0.5-second segment starting at 12 s)
- 5: movie time 12.5-13 s (0.5-second segment starting at 12.5 s)
- 6: movie time 13-13.5 s (0.5-second segment starting at 13 s)
- 7: movie time 13.5-14 s (0.5-second segment starting at 13.5 s)
- 8: movie time 14-14.5 s (0.5-second segment starting at 14 s)
- 9: movie time 14.5-15 s (0.5-second segment starting at 14.5 s)

## Files (in /app/data/)
- `train.npz`: `spikes` (int array, trials x 10 x 27), `label` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 120 trials.
- `val.npz`: same fields, a validation split. 30 trials.
- `test.npz`: `spikes` only, 50 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `label` were derived from the stimulus log, which is why `movie_time_s` is included for the labeled splits; the test trials have no stimulus log.

Load with `numpy.load(path, allow_pickle=False)`. Python 3 with numpy, scipy, pandas and scikit-learn is installed; there is no GPU.

## What to do
1. Build a decoder and predict the class of every test trial.
2. Estimate the accuracy your predictions will reach on the test trials.
3. Report any problems you find with the data.

Write `/app/output/predictions.json`:

```json
{"predictions": [<class index for test trial 0>, <test trial 1>, ...],
 "claimed_accuracy": <your estimate of your test accuracy, a number between 0 and 1>,
 "data_issues": "<what is wrong with the data, or \"none\">"}
```

`predictions` must have exactly 50 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
