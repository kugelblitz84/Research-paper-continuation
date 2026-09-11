# Block-A H2-H4 protocol extension — APPROVED

The two-task Block-A protocol extension was approved before production training
of H2, H3, or H4.

## Extension identity

- Extension version: `block_a_two_task_v1`
- Extension file: `configs/block_a_extension.yaml`
- Lock file: `configs/block_a_extension_lock.json`
- Semantic SHA-256:
  `2126d125ae9b0776a652ee55cdef8aa99a3e9ad6f46ed04ae9526203fc8c8b57`

The frozen baseline protocol remains:

- Protocol version: `full30_baseline_v1`
- Scientific configuration SHA-256:
  `1b4a63c8dda7956f0755f8a132e1d7042945a1761d55e6b3e81d9cf58868e796`

The original baseline protocol and lock are unchanged.

## Systems covered

The extension applies only to:

- H2 — Shared-Soft
- H3 — Dedicated-Hard
- H4 — Dedicated-Soft

Flat and historical H1 Shared-Hard do not receive the extension.

## Tasks and losses

H2/H3/H4 use two tasks.

### Task 1

Non-malignant versus malignant.

The existing baseline Task-1 cross-entropy is retained.

### Task 2

Melanoma / BCC / SCC.

The existing baseline Task-2 class-balanced focal loss, class weights, gamma,
task weighting and normalization are retained.

H2/H3/H4 do not use the historical H1 auxiliary T-category Task-3.

## Checkpoint selection

For H2, H3 and H4, the selected checkpoint is determined by:

`four-class endpoint validation macro-F1`

Only the frozen validation split participates in checkpoint selection.

The frozen internal test split does not participate in training, checkpoint
selection, threshold selection or Block-A development decisions.

## Routing

### H2 Shared-Soft

The four endpoint probabilities are:

`P(NM) = P1(NM)`

`P(MEL) = P1(malignant) × P2(MEL | malignant)`

`P(BCC) = P1(malignant) × P2(BCC | malignant)`

`P(SCC) = P1(malignant) × P2(SCC | malignant)`

The endpoint prediction is the argmax over these four probabilities.

### H3 Dedicated-Hard

Task 1 and Task 2 use independent complete networks.

If Task 1 predicts non-malignant, the endpoint prediction is NM.

Otherwise, the Task-2 argmax is mapped to MEL, BCC or SCC.

### H4 Dedicated-Soft

Task 1 and Task 2 use independent complete networks.

The same soft endpoint probability composition as H2 is used.

No confidence threshold or threshold tuning is introduced.

## Historical H1 preservation

H1 remains the existing three-task shared model.

Its historical checkpoint-selection rule remains:

`mean_task1_task2_task3_macro_f1`

Its Task-3 head, historical training behavior and completed EfficientNet-B0
experiment identity are unchanged.

This difference must remain disclosed when H1 is compared with the new
two-task H2/H3/H4 systems.