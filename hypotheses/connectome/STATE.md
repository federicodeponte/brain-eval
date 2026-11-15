# fly-connectome: neuron class from per-region synapse counts

Status: GO (two variants)

## Source + license
- FlyWire whole-brain connectome v783 (Dorkenwald et al. 2024; Schlegel et al. 2024, Nature). Zenodo record 10676866, CC-BY-4.0:
  per_neuron_neuropil_count_pre_783.feather (17 MB), per_neuron_neuropil_count_post_783.feather (234 MB).
- Labels: github.com/flyconnectome/flywire_annotations Supplemental_file1_neuron_annotations.tsv (super_class, cell_class, cell_type), CC-BY-4.0.

## Construction (build.py <super|mb>)
- Per neuron: input synapses per neuropil and output synapses per neuropil, top 12 regions each + "other regions" + totals.
  Neurons need >= 20 input and >= 20 output synapses.
- Prompt: class legend + 8 labeled examples (2 per class) + query. 40 test items, 10 per class, letters balanced. Prompts ~2.9k chars.
- tasks.jsonl (variant super): central brain intrinsic / descending / ascending / visual projection neuron.
- tasks_mb.jsonl (variant mb): Kenyon cell / MBON / dopaminergic neuron / antennal lobe projection neuron (fly smell-memory circuit).

## Validity (features = input and output region fractions; test neurons never in any training set)
| variant | chance | 1-NN on the 8 in-prompt shots | logistic regression on pool | kNN-5 on pool |
|---|---|---|---|---|
| super | 0.25 | 0.40 | 0.75 | 0.725 |
| mb | 0.25 | 0.55 | 0.95 | 0.875 |

## Caveats
- Region names (GNG, MB lobes, AL, LO) are informative to a model that knows fly neuroanatomy. Intended ("read the numbers + know the map"),
  but part of the signal is prior knowledge.
- FlyWire is brain only (no VNC), so ascending and descending neurons are both GNG-heavy; expected hardest pair.
- Pool and tests can contain left/right homologs of the same cell type, so pool baselines are slightly optimistic;
  1-NN on the in-prompt shots is the fair in-context reference.
- Memorization: root IDs are not shown; low risk.
