# Decode what the brain was seeing

You have spike-count recordings from 20 single units in mouse posteromedial visual area (VISpm), Allen Brain Observatory Neuropixels, while the mouse was shown drifting gratings (2 s each, motion directions, mixed temporal frequencies).

Each trial is a 20 x 8 matrix: 20 units by 8 time bins of 250 ms starting at stimulus onset. The task is to decode which of 6 stimulus classes was shown on each trial:

- 0: grating direction 0 deg
- 1: grating direction 45 deg
- 2: grating direction 90 deg
- 3: grating direction 135 deg
- 4: grating direction 225 deg
- 5: grating direction 315 deg

## Files (in /app/data/)
- `train_labeled.npz`: `spikes` (int array, trials x 20 x 8), `label` (class index per trial), `orientation_deg` (the grating direction/orientation in degrees shown on the trial, from the stimulus log). 275 trials.
- `val_labeled.npz`: same fields, a validation split. 78 trials.
- `test_unlabeled.npz`: `spikes` only, 114 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `label` were derived from the stimulus log, which is why `orientation_deg` is included for the labeled splits; the test trials have no stimulus log.

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

`predictions` must have exactly 114 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
