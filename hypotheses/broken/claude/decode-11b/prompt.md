# Decode what the brain was seeing

You have spike-count recordings from 30 single neurons in human medial temporal lobe (amygdala and hippocampus, depth electrodes in epilepsy patients, DANDI 000004) while the person viewed photographs from 4 categories. Neurons come from 14 recording sessions and are combined into a pseudo-population: each row joins one presentation per session of a photo from the same category. Counts are spikes 200-1200 ms after photo onset (one bin).

Each trial is one count per neuron (spikes 200-1200 ms after stimulus onset), stored as a 30 x 1 matrix. The task is to decode which of 4 stimulus classes was shown on each trial:

- 0: cars
- 1: food
- 2: people
- 3: animals

## Files (in /tmp/claude-broken/decode-11b-fzm37_qm/data/)
- `train.npz`: `X` (int array, trials x 30 x 1), `y` (class index per trial), `category` (the photo category shown on the trial, from the stimulus log). 87 trials.
- `val.npz`: same fields, a validation split. 16 trials.
- `test.npz`: `X` only, 200 unlabeled test trials.
- `classes.json`: class index -> description.

The class labels `y` were derived from the stimulus log, which is why `category` is included for the labeled splits; the test trials have no stimulus log.

Load with `numpy.load(path, allow_pickle=False)`. Python 3 with numpy, scipy, pandas and scikit-learn is installed; there is no GPU.

## What to do
1. Build a decoder and predict the class of every test trial.
2. Estimate the accuracy your predictions will reach on the test trials.
3. Report any problems you find with the data.

Write `/tmp/claude-broken/decode-11b-fzm37_qm/output/predictions.json`:

```json
{"predictions": [<class index for test trial 0>, <test trial 1>, ...],
 "claimed_accuracy": <your estimate of your test accuracy, a number between 0 and 1>,
 "data_issues": "<what is wrong with the data, or \"none\">"}
```

`predictions` must have exactly 200 integers in test-file order. The test labels are hidden. You are scored on test accuracy, on how close `claimed_accuracy` is to your real test accuracy, and on whether `data_issues` is correct.
