# protein-mutations: pick the most damaging of 4 single-point mutations

Status: GO

## Source + license
- ProteinGym v1.3 DMS substitutions (Notin et al., NeurIPS 2023), marks.hms.harvard.edu/proteingym (openly released; repo MIT license). Cite ProteinGym + each assay paper.
- Assays (less famous proteins, 2015-2023 DMS): MLAC_ECOLI_MacRae_2023 (MlaC lipid transporter), RNC_ECOLI_Weeks_2023 (RNase III),
  ESTA_BACSU_Nutschel_2020 (estA lipase, thermostability), A0A247D711_LISMN_Stadelmann_2021 (anti-CRISPR AcrIIA4).
  Files: raw/assays/*.csv, raw/DMS_substitutions.csv (43 MB zip deleted after extraction; re-download from ProteinGym_v1.3 URL).

## Construction (build.py)
- 10 items per assay = 40. Each item: wild-type sequence + 4 single substitutions at 4 different positions.
  Exactly one is in the bottom 10% of that assay's DMS_score (damaging); the other three are in the top 50% (tolerated).
- Letters balanced 10/10/10/10. Prompts < 700 chars. meta: DMS scores, quantile cutoffs, BLOSUM62 per option.
- Rebuild: `../_tools/.v4/bin/python -I build.py`

## Validity (no baseline sees the query labels)
| method | acc |
|---|---|
| chance | 0.25 |
| BLOSUM62: pick most negative substitution score | 0.50 |
| hydropathy change + proline penalty | 0.425 |
| position sensitivity: mean DMS of OTHER variants at that position (query variants excluded) | 0.925 |

BLOSUM62 per assay: MLAC 6/10, RNC 6/10, ESTA 3/10, AcrIIA4 5/10.

## Caveats
- The position-mean baseline uses the rest of the same assay: it proves the label is recoverable (position-driven), it is not a fair
  "no experiment" comparison. The fair prior-knowledge reference is BLOSUM62 at 0.50. ESM not run (CPU + disk budget; AX41 home at 96%).
- BLOSUM ties break to the first listed option (letters are balanced, so no inflation).
- Memorization: these DMS maps are public, but per-variant scores of these proteins are unlikely memorized. A model can still use real
  biochemistry (buried hydrophobics, signal peptide, catalytic residues), which is the point. MlaC sequence includes its signal peptide.
