# Notebook workflow

Run the active notebooks in numeric order. Training and development finish before
final frozen-test evaluation and comparison.

| Order | Notebook | Purpose |
|---|---|---|
| 00 | `00_environment_and_protocol_audit.ipynb` | Environment and frozen protocol |
| 01 | `01_dataset_and_split_verification.ipynb` | Frozen dataset/splits |
| 02 | `02_flat_training.ipynb` | Flat |
| 03 | `03_shared_hierarchical_training.ipynb` | Existing Shared-Hard |
| 04 | `04_shared_soft_training.ipynb` | Shared-Soft |
| 05 | `05_dedicated_hard_training.ipynb` | Dedicated-Hard |
| 06 | `06_dedicated_soft_training.ipynb` | Dedicated-Soft |
| 07 | `07_block_a_status.ipynb` | All 35 training/validation cells and validation selection |
| 08 | `08_final_internal_test_evaluation.ipynb` | Deliberate final test evaluation |
| 09 | `09_final_35_cell_comparison.ipynb` | Consolidated research tables and figures |

The original B0-only notebooks 04 and 05 are preserved byte-for-byte in `archive/`,
including their saved outputs. They are historical records outside the active
workflow. **The archived original evaluation notebook contains RUN_EVALUATION=True
and its original VM path; do not Run All on that historical record.** Use the safe
new notebook 08 instead. Earlier saved notebooks may mention the old next step;
this workflow index defines the current order without rewriting historical outputs.

## Final evaluation

Notebook 08 starts with `RUN_EVALUATION=False`, `DEVELOPMENT_FROZEN=False`, and an
empty explicit `RUN_IDS` list. Default Run All only displays status/evidence.
The execution helper requires all 35 training cells and their endpoint validation
results to be complete before final inference. Select the desired run IDs and opt
in only after development decisions are frozen. Existing test results are skipped;
missing or mismatched historical evidence is reviewed/restored, not regenerated.
The dataset root comes from `SKIN_CANCER_DATA_ROOT`.

## Final comparison and illustration

Notebook 09 reads saved endpoint results for `SPLIT="test"`. It neither trains nor
infers. Its main illustration combines:

- all 35 endpoint macro-F1 scores;
- all 35 balanced accuracies;
- all 35 SCC F1 scores;
- matched hierarchy-versus-Flat macro-F1 differences.

Supporting figures show all four class F1 matrices and a 7-by-5 confusion-matrix
panel. Tables preserve overall metrics, per-class metrics/supports, routing
sensitivity/oracle gaps, means, medians, deltas against Flat and H1, coverage and
checkpoint/prediction/configuration provenance.

Only one unambiguous result identity per cell is included. If a cell has multiple
saved identities, specify `RESULT_CHOICES={run_id: result_id}` explicitly. The
report rejects unmatched manifest/sample-count cohorts. Missing values remain
empty/grey and an incomplete report is visibly labelled INCOMPLETE. No real result
has been invented to populate the paper illustration.

Set `EXPORT_PAPER_ARTIFACTS=True` to write full-precision CSVs, 300-dpi PNGs, vector
SVG/PDFs, a caption and a hashed report manifest to
`results/final/<split>_<content_digest>/`. Repeated identical exports verify/reuse
that bundle. Printed labels use three decimals, but stored metrics retain precision.

These are descriptive figures; they do not invent confidence intervals or p-values.
Paired inference requires saved per-image predictions and a predefined comparison
and multiplicity plan. H1 retains its historical auxiliary third task/selection
rule. Test comparisons do not change the validation-selected hierarchy or trigger
further experimental blocks. The H2-H4 protocol extension is approved as `block_a_two_task_v1`.
H2/H3/H4 use four-class endpoint validation macro-F1 for checkpoint selection;
historical H1 retains its original three-task selection rule.
