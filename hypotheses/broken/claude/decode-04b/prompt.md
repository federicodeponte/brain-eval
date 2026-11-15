# Decode what the brain was seeing

You have spike-count recordings from 25 single units in mouse primary visual cortex (VISp), Allen Brain Observatory Neuropixels, while the mouse watched a natural movie that was replayed 10 times.

Each trial is a 25 x 10 matrix: 25 units (rows) by 10 consecutive time bins of 100 ms (columns) starting at stimulus onset. The task is to decode which of 10 stimulus classes was shown on each trial:

- 0: movie seconds 60-61 (1-second segment starting at movie time 60 s)
- 1: movie seconds 61-62 (1-second segment starting at movie time 61 s)
- 2: movie seconds 62-63 (1-second segment starting at movie time 62 s)
- 3: movie seconds 63-64 (1-second segment starting at movie time 63 s)
- 4: movie seconds 64-65 (1-second segment starting at movie time 64 s)
- 5: movie seconds 65-66 (1-second segment starting at movie time 65 s)
- 6: movie seconds 66-67 (1-second segment starting at movie time 66 s)
- 7: movie seconds 67-68 (1-second segment starting at movie time 67 s)
- 8: movie seconds 68-69 (1-second segment starting at movie time 68 s)
- 9: movie seconds 69-70 (1-second segment starting at movie time 69 s)

## Files (in /tmp/claude-broken/decode-04b-skcz1qm7/data/)
- `train.npz`: `X` (int array, trials x 25 x 10), `y` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 60 trials.
- `val.npz`: same fields, a validation split. 10 trials.
- `test.npz`: `X` only, 30 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `y` were derived from the stimulus log, which is why `movie_time_s` is included for the labeled splits; the test trials have no stimulus log.

Load with `numpy.load(path, allow_pickle=False)`. Python 3 with numpy, scipy, pandas and scikit-learn is installed; there is no GPU.

## What to do
1. Build a decoder and predict the class of every test trial.
2. Estimate the accuracy your predictions will reach on the test trials.
3. Report any problems you find with the data.

Write `/tmp/claude-broken/decode-04b-skcz1qm7/output/predictions.json`:

```json
{"predictions": [<class index for test trial 0>, <test trial 1>, ...],
 "claimed_accuracy": <your estimate of your test accuracy, a number between 0 and 1>,
 "data_issues": "<what is wrong with the data, or \"none\">"}
```

`predictions` must have exactly 30 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
