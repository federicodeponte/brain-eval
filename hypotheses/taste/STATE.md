# TASTE - GO

- Data: ChemTastesDB v2.1, Rojas et al. 2022, Zenodo record 15051366, CC-BY-4.0. File: _raw/taste/ChemTastesDB_database.xlsx (4078 rows).
- Classes used: Sweetness / Bitterness / Umaminess. Canonicalized with RDKit; molecules labelled with more than one of these classes dropped; deduped. 2954 molecules (1547 bitter, 1187 sweet, 220 umami).
- Items: 40 (14 sweet=A, 13 bitter=B, 13 umami=C), fixed choices A sweet / B bitter / C umami. SMILES only, name withheld (meta keeps name/CID/ChemTastesDB ID). Obscurity filter: PubChem CID required, PubChem synonym count <= 15, famous-name blocklist (sucrose, aspartame, caffeine, quinine, ...). Seeded (random_state 3/5), script taste/build.py.
- Chance: 33.3% (max-class guess on these items 35%).
- Baseline (Morgan r2 2048 bits, trained on all non-item tastants, n=2914): logistic regression 80.0%, kNN-5 72.5%.
- Strict baseline (train molecules with Tanimoto >= 0.6 to any item removed, n=2625): LR 72.5%, kNN-5 70.0%. Clearly above chance -> structure carries the signal.
- Caveats / leakage:
  - Umami items are mostly peptides / glutamate-like acids (8/13 umami items contain an amide bond); a chemist (or model) can read off "Glu/Asp-rich peptide -> umami". Bitter peptides exist too (example_1 is a bitter Arg-Pro-Gly-Phe peptide), which keeps it non-trivial.
  - ChemTastesDB is compiled from published literature; low-synonym molecules are still in papers, so memorization is possible but unlikely per item.
  - Taste labels are panel reports at given concentrations; some compounds are sweet/bitter dual (those dropped).
- Examples: taste/examples/example_1.txt, example_2.txt.
