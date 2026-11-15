# Decode what the brain was seeing

You have spike-count recordings from 17 single units in mouse anterolateral visual area (VISal), Allen Brain Observatory Neuropixels, while the mouse watched natural movie three (120 s clip), replayed 10 times.

Each trial is a 5 x 17 matrix: 5 time bins of 100 ms by 17 units starting at stimulus onset. The task is to decode which of 9 stimulus classes was shown on each trial:

- 0: movie time 42-42.5 s (0.5-second segment starting at 42 s)
- 1: movie time 42.5-43 s (0.5-second segment starting at 42.5 s)
- 2: movie time 43-43.5 s (0.5-second segment starting at 43 s)
- 3: movie time 43.5-44 s (0.5-second segment starting at 43.5 s)
- 4: movie time 44-44.5 s (0.5-second segment starting at 44 s)
- 5: movie time 44.5-45 s (0.5-second segment starting at 44.5 s)
- 6: movie time 45-45.5 s (0.5-second segment starting at 45 s)
- 7: movie time 45.5-46 s (0.5-second segment starting at 45.5 s)
- 8: movie time 46-46.5 s (0.5-second segment starting at 46 s)

## Files (in /app/data/)
- `train_labeled.npz`: `counts` (int array, trials x 5 x 17), `label` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 45 trials.
- `val_labeled.npz`: same fields, a validation split. 18 trials.
- `test_unlabeled.npz`: `counts` only, 27 unlabeled test trials.
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

`predictions` must have exactly 27 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
