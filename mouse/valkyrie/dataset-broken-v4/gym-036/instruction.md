# Decode what the brain was seeing

You have spike-count recordings from 29 single units in mouse posteromedial visual area (VISpm), Allen Brain Observatory Neuropixels, while the mouse was shown flashed natural photographs (250 ms each, random order).

Each trial is a 29 x 2 matrix: 29 units by 2 time bins of 125 ms starting at stimulus onset. The task is to decode which of 5 stimulus classes was shown on each trial:

- 0: natural scene image #25 of the Allen natural-scenes set
- 1: natural scene image #57 of the Allen natural-scenes set
- 2: natural scene image #63 of the Allen natural-scenes set
- 3: natural scene image #67 of the Allen natural-scenes set
- 4: natural scene image #113 of the Allen natural-scenes set

## Files (in /app/data/)
- `train.npz`: `X` (int array, trials x 29 x 2), `stim_class` (class index per trial), `image_id` (the Allen natural-scenes image number shown on the trial, from the stimulus log). 135 trials.
- `val.npz`: same fields, a validation split. 50 trials.
- `test.npz`: `X` only, 65 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `stim_class` were derived from the stimulus log, which is why `image_id` is included for the labeled splits; the test trials have no stimulus log.

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

`predictions` must have exactly 65 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
