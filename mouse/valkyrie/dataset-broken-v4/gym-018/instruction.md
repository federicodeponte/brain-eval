# Decode what the brain was seeing

You have spike-count recordings from 28 single units in mouse posteromedial visual area (VISpm), Allen Brain Observatory Neuropixels, while the mouse watched natural movie one (30 s clip), replayed 20 times.

Each trial is a 10 x 28 matrix: 10 time bins of 100 ms by 28 units starting at stimulus onset. The task is to decode which of 6 stimulus classes was shown on each trial:

- 0: movie time 6-7 s (1-second segment starting at 6 s)
- 1: movie time 7-8 s (1-second segment starting at 7 s)
- 2: movie time 8-9 s (1-second segment starting at 8 s)
- 3: movie time 9-10 s (1-second segment starting at 9 s)
- 4: movie time 10-11 s (1-second segment starting at 10 s)
- 5: movie time 11-12 s (1-second segment starting at 11 s)

## Files (in /app/data/)
- `fit.npz`: `counts` (int array, trials x 10 x 28), `label` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 72 trials.
- `holdout.npz`: same fields, a validation split. 18 trials.
- `eval.npz`: `counts` only, 30 unlabeled test trials.
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

`predictions` must have exactly 30 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
