# Baseline reconstruction build handover

**Build complete. Stop before real training.** No Azure run, raw-image training,
pretrained-weight download, or new scientific reproduction result is claimed.
The implementation is ready for researcher review and subsequent Azure preflight.

## 1. Repository tree

```text
README.md
pyproject.toml, requirements.txt, requirements-dev.txt
configs/
  protocol.yaml, protocol_lock.json
  experiments/flat/          7 YAML configs
  experiments/shared_hard/   7 YAML configs
data/manifests/
  isic2019_train_val_test_split_seed42.csv
  emb_stage03_dermoscopic_split_seed42.csv
  isic2019_split_groups_seed42.csv
  manifest_lock.json
src/
  config.py, data.py, transforms.py, losses.py
  models/backbones.py, models/systems.py
  training.py, artifacts.py, reproducibility.py
  evaluation.py, routing.py, metrics.py, statistics.py, reporting.py
notebooks/                  00 through 05
experiments/
  experiment_registry.csv  header only; no real runs
  runs/                    per-run snapshots/history/logs/checkpoint refs
  evaluations/             per-evaluation records
models/checkpoints/        immutable best.pt/last.pt generations
results/
  historical/              source-labelled legacy data, never new-run results
  predictions/, tables/, figures/
tests/                     33 CPU tests
 tools/                    import/reference parity/statistics replay/notebook QA
 docs/                     audit, unresolved items, setup, recovery, continuation
 validation/               executed verification evidence
 reference/historical/     ignored optional ZIP evidence; never a runtime dependency
```

## 2. Historical protocol discovered

The source-to-implementation parameter table is in
[HISTORICAL_PROTOCOL_AUDIT.md](HISTORICAL_PROTOCOL_AUDIT.md), written before the
experimental implementation. The selected reference is the Phase06/Phase11/Phase02
Flat CE family and Phase03 Shared three-task protocol generalized by Phase06.
Older standalone cascades and Flat focal candidates are not substituted.

Seed 42; ImageNet initialization; all backbones at 224x224; historical moderate
train-only augmentation; deterministic resize256/center-crop224 evaluation;
Dropout .2; batch64/workers4; AdamW lr3e-4/wd1e-4; cosine T_max30/min_lr1e-6;
maximum30 epochs; patience7; strict validation improvement. Flat loss is CE.
Shared losses are task1 CE, task2 class-balanced focal, task3 weighted CE, with
historical weights and equal active-task weighting. Flat validation uses CUDA AMP;
shared validation is full precision. Test is structurally separate from training.

## 3. Discrepancies resolved from source

- The pasted conceptual two-head outline omits the actual auxiliary third head.
  The historical comparison trained **three tasks**, concatenated 594 T-category
  training images and selected by mean task1/2/3 validation macro-F1. All are retained.
  Only Stage1/2 determine the four-class hard endpoint.
- Flat MobileNetV3 has a native Linear960->1280/Hardswish projection before the final
  Dropout/Linear. It is retained; shared MobileNet uses 960 pooled features directly.
- The broad document roadmaps propose later soft/dedicated/ensemble/head experiments.
  Those proposals do not override the user's current baseline-only scope.
- The final seven-backbone/HIBA synthesis cited by the documents is absent from the
  ZIP. Their rounded seven-backbone table is labelled document-reported. B0's stored
  comparative predictions are present and were independently replayed.

## 4. Exact architectures implemented

| Config identifier | Encoder feature dimension | Flat parameters | Shared parameters |
|---|---:|---:|---:|
| efficientnet_b0 | 1280 | 4,012,672 | 4,020,358 |
| densenet121 | 1024 | 6,957,956 | 6,964,106 |
| densenet169 | 1664 | 12,491,140 | 12,501,130 |
| resnet50 | 2048 | 23,516,228 | 23,528,522 |
| mobilenet_v3_large | 960 | 4,207,156 | 2,981,562 |
| efficientnet_b2 | 1408 | 7,706,630 | 7,715,084 |
| efficientnet_b3 | 1536 | 10,702,380 | 10,711,602 |

Counts were measured on the reconstructed models and matched the historical models.

## 5. Flat architecture

Central backbone factory -> native pooling -> pooled feature vector ->
Dropout(.2) -> Linear4. DenseNet includes its historical final ReLU before pooling.
MobileNet additionally retains the historical Linear/Hardswish projection.

## 6. Shared-Hard architecture

Exactly one encoder and one pooled feature vector feed three independent
Dropout(.2)/Linear heads with 2/3/5 logits. Masking selects valid samples before each
criterion. Loss is the weighted sum of active per-task mean losses divided by the
active weight sum. The third head is auxiliary to this four-class endpoint.

Hard routing: Stage1 argmax NM -> endpoint0; otherwise Stage2 argmax+1.
Oracle routing: true NM -> endpoint0; true malignant -> Stage2 argmax+1.
Oracle is a truth-assisted diagnostic, not a deployable model.

## 7. Configs created

For every identifier in section4, both paths exist:

```text
configs/experiments/flat/<identifier>.yaml
configs/experiments/shared_hard/<identifier>.yaml
```

All14 select `configs/protocol.yaml`. Its digest lock rejects scientific drift,
including in-memory changes made inside a notebook before a real training call.
B0 is the only backbone initially allowed through the real training gate.

## 8. Notebooks created

1. `00_environment_and_protocol_audit.ipynb`
2. `01_dataset_and_split_verification.ipynb`
3. `02_flat_training.ipynb`
4. `03_shared_hierarchical_training.ipynb`
5. `04_internal_test_evaluation.ipynb`
6. `05_flat_vs_hierarchical_comparison.ipynb`

Each declares objective/config/input/output/mode, loads its own inputs from disk,
and ends with a summary/next step. All reusable scientific logic is under `src/`.
Training, real evaluation and reproduction acceptance default to disabled.

## 9-10. Tests executed and actual results

| Verification | Actual result | Evidence |
|---|---|---|
| CPU pytest suite | 33 passed | validation/pytest.xml |
| Fourteen config-driven model paths | Correct224px output dimensions; one encoder call | pytest + historical_parity.json |
| Historical model parity | Exact tensor equality for all14 models in eval and train modes; equal parameter counts | validation/historical_parity.json |
| Transform parity | Train and evaluation bitwise equal to historical transforms under matched RNG | same |
| Actual B0 backward smoke | Flat and shared optimizer steps on synthetic inputs passed | pytest |
| Synthetic orchestration | Training -> frozen checkpoint -> predictions -> paired statistics passed | pytest |
| Interrupted resume | Both families matched uninterrupted tiny-model weights/history/scheduler with workers0 | pytest |
| Frozen manifests | Exact hashes/counts and disjoint groups; no cross-source split conflicts | validation/manifest_audit.json |
| Checkpoint preservation | Save/load roundtrip, overwrite refusal, hash tamper and config drift checks passed | pytest |
| Historical statistics replay | All10000 resamples; reproduced stored interval and exact McNemar | validation/historical_statistics_replay.json |
| Static checks | Ruff check and format check passed | commands recorded below |
| Reporting figure smoke | Generated and visually inspected; no clipped labels | validation/historical_replay_figure.png |
| Editable package install | Succeeded in workspace .venv | validation/installed_packages.json |

Historical B0 replay: Flat macro-F1 0.6192224685168973, Shared-Hard
0.5685909456725847; difference -0.050631522844312604; 95% paired interval
[-0.07599324143900181, -0.02482655293932654]; exact McNemar p=2.4911763796442693e-13.
These are **recomputed historical predictions, not newly retrained results**.

Executed commands:

```text
.venv/Scripts/python.exe -m pytest -q --junitxml=validation/pytest.xml
.venv/Scripts/python.exe -m ruff check src tests tools
.venv/Scripts/python.exe -m ruff format --check src tests tools
.venv/Scripts/python.exe tools/verify_historical_parity.py
.venv/Scripts/python.exe tools/replay_historical_statistics.py
.venv/Scripts/python.exe tools/validate_notebooks.py
.venv/Scripts/python.exe tools/import_reference.py F:/hobby/research_paper.zip
```

## 11. Notebook validation

**6/6 passed schema validation, Python compilation and fresh-kernel execution**
with default safe controls. Executed copies are under `validation/executed/`;
the machine-readable summary is `validation/notebook_validation.json`.
Training/evaluation branches were tested separately with synthetic orchestration;
real dataset-dependent notebook branches and CUDA AMP were not executed.
Local Jupyter emitted Windows event-loop/TCP transport warnings, but no cell failed.

## 12. Unresolved protocol/evidence items

See [UNRESOLVED_PROTOCOL_ITEMS.md](UNRESOLVED_PROTOCOL_ITEMS.md): no raw images or
historical checkpoints in ZIP; missing final seven-backbone synthesis; no original
complete GPU environment capture; no specified retraining acceptance tolerance;
multiworker augmentation resume is not bitwise guaranteed. No Git commit has been
created in this workspace; environment provenance currently reports git_commit=null.

The two scientific discrepancies above have explicit source-backed implementation
decisions. A genuinely two-task shared design would be a later separately labelled
experiment, not an unannounced replacement for the preserved baseline.

## 13. Dataset setup

Use the original images and the exact frozen manifests. Set `SKIN_CANCER_DATA_ROOT`
to a directory containing:

```text
data/raw/isic2019/images/ISIC_2019_Training_Input/*.jpg
data/raw/emb/images/isic/*.jpg
```

Run notebook01, then enable its train/validation image hash check. Do not infer
eligibility from filenames or regenerate splits. The shared auxiliary cohort has
594/127/127 train/validation/test images; the endpoint cohort has17124/3668/3668.

## 14. Checkpoint storage

`models/checkpoints/<run_id>/epoch_<n>_<generation>/best.pt` and `last.pt`.
`experiments/runs/<run_id>/checkpoints.json` records latest/best paths and hashes;
`frozen_checkpoint.json` freezes the selected best model at completion.
All previous checkpoint generations are retained. **Git does not preserve ignored
checkpoints.** Back up actual files plus their run/prediction/evaluation records.
See [recovery instructions](CHECKPOINTS_AND_RECOVERY.md).

## 15. Azure setup

See [AZURE_SETUP.md](AZURE_SETUP.md). Python3.11.9 and the historical torch2.13.0 /
torchvision0.28.0 pair are pinned. Install a compatible CUDA wheel/driver, verify
`nvidia-smi` and `torch.cuda.is_available()`, install the Jupyter kernel, configure
durable storage and image paths, then rerun CPU tests and image preflight on the VM.
Actual CUDA/environment verification remains pending; nothing was provisioned here.

## 16. Recommended execution order

Build review -> Azure environment/image preflight -> notebook02 B0 Flat ->
notebook03 B0 Shared-Hard -> notebook04 frozen test predictions -> notebook05 paired
comparison and researcher reproduction review -> remaining six backbone pairs.
Future systems start at notebook06 after freezing this baseline.

## 17. Exact next action

**Review this handover and authorize the real Azure EfficientNet-B0 reproduction
pair when ready.** After authorization, configure the dataset on Azure, run00/01,
then change only `RUN_TRAINING=True` in notebook02 while keeping
`CONFIG_PATH="configs/experiments/flat/efficientnet_b0.yaml"`. Complete Flat before
launching the B0 Shared-Hard notebook. The remaining six pairs stay locked until
notebook05 records an explicit accepted reproduction review.
