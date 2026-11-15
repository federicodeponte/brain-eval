# Decode what the brain was seeing

You have spike-count recordings from 18 single units in mouse posteromedial visual area (VISpm), Allen Brain Observatory Neuropixels, while the mouse watched natural movie three (120 s clip), replayed 10 times.

Each trial is a 10 x 18 matrix: 10 time bins of 100 ms by 18 units starting at stimulus onset. The task is to decode which of 6 stimulus classes was shown on each trial:

- 0: movie time 80-81 s (1-second segment starting at 80 s)
- 1: movie time 81-82 s (1-second segment starting at 81 s)
- 2: movie time 82-83 s (1-second segment starting at 82 s)
- 3: movie time 83-84 s (1-second segment starting at 83 s)
- 4: movie time 84-85 s (1-second segment starting at 84 s)
- 5: movie time 85-86 s (1-second segment starting at 85 s)

## Files (in /app/data/)
- `fit.npz`: `counts` (int array, trials x 10 x 18), `stim_class` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 30 trials.
- `holdout.npz`: same fields, a validation split. 12 trials.
- `eval.npz`: `counts` only, 18 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `stim_class` were derived from the stimulus log, which is why `movie_time_s` is included for the labeled splits; the test trials have no stimulus log.

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

`predictions` must have exactly 18 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
