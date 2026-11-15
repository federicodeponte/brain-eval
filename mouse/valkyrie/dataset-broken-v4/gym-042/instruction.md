# Decode what the brain was seeing

You have spike-count recordings from 29 single units in mouse rostrolateral visual area (VISrl), Allen Brain Observatory Neuropixels, while the mouse watched natural movie three (120 s clip), replayed 10 times.

Each trial is a 29 x 20 matrix: 29 units by 20 time bins of 50 ms starting at stimulus onset. The task is to decode which of 9 stimulus classes was shown on each trial:

- 0: movie time 104-105 s (1-second segment starting at 104 s)
- 1: movie time 105-106 s (1-second segment starting at 105 s)
- 2: movie time 106-107 s (1-second segment starting at 106 s)
- 3: movie time 107-108 s (1-second segment starting at 107 s)
- 4: movie time 108-109 s (1-second segment starting at 108 s)
- 5: movie time 109-110 s (1-second segment starting at 109 s)
- 6: movie time 110-111 s (1-second segment starting at 110 s)
- 7: movie time 111-112 s (1-second segment starting at 111 s)
- 8: movie time 112-113 s (1-second segment starting at 112 s)

## Files (in /app/data/)
- `train.npz`: `X` (int array, trials x 29 x 20), `y` (class index per trial), `movie_time_s` (the movie time in seconds at which the trial's segment starts, from the stimulus log). 45 trials.
- `val.npz`: same fields, a validation split. 18 trials.
- `test.npz`: `X` only, 27 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `y` were derived from the stimulus log, which is why `movie_time_s` is included for the labeled splits; the test trials have no stimulus log.

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
