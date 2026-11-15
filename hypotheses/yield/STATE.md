# REACTION YIELD - GO

- Data: Ahneman, Estrada, Lin, Dreher, Doyle, Science 2018 (Buchwald-Hartwig HTE, 15 aryl halides x 4 ligands x 3 bases x 23 isoxazole additives, p-toluidine, Pd G3 precatalysts). File: doylelab/rxnpredict data_table.csv (MIT license on repo). After dropping additive-free controls: 4132 reactions.
- Items: 40. Each item = one aryl halide (all 15 used, 2-3 items each) and 4 distinct wells (ligand, base, additive) from the screen. Constraints: best yield >= 15%, best beats 2nd by >= 10 points, >= 2 distinct ligands and >= 2 distinct bases among options. No reaction reused across items (160 wells held out). Answer letters 10/10/10/10. Seed 11, script yield/build.py. meta.yields holds all 4 measured yields.
- Chance: 25%.
- Baseline RF (300 trees, one-hot of halide/ligand/base/additive), trained on all non-item reactions: 90.0%.
- Strict baseline, leave-aryl-halide-out (train excludes every reaction with the item halide): RF 77.5%.
- Additive main-effects heuristic (sum of training-mean yield of ligand + base + additive): 82.5% random split / 80% halide-out. So the items are mostly solvable from component main effects.
- Caveats / leakage:
  - The dataset and its main findings (MTBD best base on average, XPhos weakest ligand, some isoxazole additives poison the catalyst) are widely published and used as an ML benchmark; a model may know these trends. That is legitimate chemistry knowledge, but report it as "knows the 2018 paper trends" rather than "senses reactivity".
  - Yields are single HTE measurements; the 10-point gap filter keeps noise from deciding the answer.
  - For a harder variant: same ligand and base across all 4 options (additive-only choice) - not built.
- Examples: yield/examples/example_1.txt, example_2.txt.
