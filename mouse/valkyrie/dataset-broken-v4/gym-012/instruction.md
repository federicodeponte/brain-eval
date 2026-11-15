# Decode what the brain was seeing

You have spike-count recordings from 27 single units in mouse posteromedial visual area (VISpm), Allen Brain Observatory Neuropixels, while the mouse watched natural movie one (30 s clip), replayed 20 times.

Each trial is a 27 x 10 matrix: 27 units by 10 time bins of 100 ms starting at stimulus onset. The task is to decode which of 9 stimulus classes was shown on each trial:

- 0: movie time 19-20 s (1-second segment starting at 19 s)
- 1: movie time 20-21 s (1-second segment starting at 20 s)
- 2: movie time 21-22 s (1-second segment starting at 21 s)
- 3: movie time 22-23 s (1-second segment starting at 22 s)
- 4: movie time 23-24 s (1-second segment starting at 23 s)
- 5: movie time 24-25 s (1-second segment starting at 24 s)
- 6: movie time 25-26 s (1-second segment starting at 25 s)
- 7: movie time 26-27 s (1-second segment starting at 26 s)
- 8: movie time 27-28 s (1-second segment starting at 27 s)

## Files (in /app/data/)
- `fit.npz`: `X` (int array, trials x 27 x 10), `label` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 108 trials.
- `holdout.npz`: same fields, a validation split. 27 trials.
- `eval.npz`: `X` only, 45 unlabeled test trials.
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

`predictions` must have exactly 45 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
