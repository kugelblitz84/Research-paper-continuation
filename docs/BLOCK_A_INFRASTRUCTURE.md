# Block-A infrastructure and pending protocol decision

The repository now represents F (Flat), H1 (Shared-Hard), H2 (Shared-Soft), H3
(Dedicated-Hard), and H4 (Dedicated-Soft), across DenseNet121, DenseNet169,
ResNet50, MobileNetV3-Large, EfficientNet-B0, EfficientNet-B2, and EfficientNet-B3.
These are 35 system cells, or 49 independent network instances when each cell is
trained independently. H1 retains its existing third auxiliary T-category head.

**Production H2-H4 training is protocol-enabled.** The approved two-task extension is
`configs/block_a_extension.yaml`, version `block_a_two_task_v1`, with semantic digest
`2126d125ae9b0776a652ee55cdef8aa99a3e9ad6f46ed04ae9526203fc8c8b57`.
H2/H3/H4 select checkpoints by four-class endpoint validation macro-F1.
The frozen baseline protocol and historical H1 behavior remain unchanged.

## Configuration and execution

`configs/experiments/<system>/<backbone>.yaml` resolves through the existing
`src.config.load_config`. The seven-family encoder factory remains
`src/models/backbones.py`. Run IDs follow `<system>_<backbone>_seed42`; the resolved
configuration digest is checked separately. Other seeds require explicit protocol
versioning. Flat/H1 configurations retain their original hashes.

Status and backfill are safe without raw data, GPU, or VM checkpoints:

```text
python -m src.run_block_a --status
python -m src.results_registry rebuild
```

The first production verification experiment is:

```text
python -m src.run_experiment --system shared_soft --backbone efficientnet_b0 --seed 42
```

Set `SKIN_CANCER_DATA_ROOT` first. The CLI is an explicit training request; the
Python `run_experiment` API defaults to `allow_training=False`. Training notebooks
04-06 default to `RUN_TRAINING=False`; notebook 07 defaults to `REBUILD_INDEX=False`.
Notebook 08 performs explicitly requested final evaluation after all training/validation;
notebook 09 consolidates saved results into paper tables and figures. Both are safe by default.
See [the notebook workflow](../notebooks/README.md).
No runner automatically evaluates test data. The original other-backbone
reproduction-review gate remains in force.

A later system-wide command is `python -m src.run_block_a --system shared_soft`.
It skips completed experiments, reports failed/incomplete cells, and aborts on
scientific incompatibility. `--all` is available but was not executed. Explicit
`--resume` is only for a matching interrupted run with canonical configuration and
checkpoint metadata. No folders are renamed, checkpoints replaced, or stale locks
removed automatically. Registry-only completion is protected even without VM files.

## Artifact flow

Configuration -> model -> existing training loop -> validation checkpoint selection
-> immutable selected checkpoint -> saved endpoint validation probabilities/metrics
-> append-only experiment events -> checkpoint/result indexes -> Block-A aggregation.

- `experiments/experiment_registry.csv`: unchanged append-only event schema.
- `experiments/runs/<run_id>/`: resolved YAML, environment, history, logs, summary,
  last/best checkpoint pointers and frozen checkpoint manifest.
- `models/checkpoints/<run_id>/epoch_.../`: immutable best/last tensors. Dedicated
  systems additionally export `task1_best.pt` and `task2_best.pt` from the selected
  joint epoch. The joint checkpoint holds both models and optimizer/resume state.
- `experiments/evaluations/<run_id>_validation_<checkpoint_hash_prefix>/evaluation.json`:
  selected-checkpoint endpoint, per-class, confusion-matrix and diagnostic metrics.
- `results/predictions/`: full precision per-image CSVs. Hierarchical predictions
  include conditional task probabilities, hard/oracle diagnostic decisions, and
  normalized endpoint probabilities. For hard systems the endpoint distribution
  is hard-gated; for soft systems it is the unconditional product distribution.
- `experiments/checkpoint_registry.csv`: derived component-aware index; historical
  checkpoint generations are retained and unknown hashes stay empty.
- `results/master_results.csv`: one row per immutable run/split/mode/prediction/
  configuration/checkpoint identity. Conflicting metrics for identical inputs fail
  loudly. Distinct scientific identities are retained. JSON metric fields retain
  class information for binary and subtype diagnostics as well as endpoint metrics.
- `results/block_a/status.csv`, `validation_matrix.csv`, `system_summary.csv`,
  `selection.json`: derived development-only summaries. Partial means/medians and
  matched delta counts are labelled; selection requires all 35 eligible cells.

Rebuild reads only canonical JSON/YAML/CSV metadata and verifies prediction hashes
when prediction files are present. It never unpickles a model, trains, infers, or
recomputes missing metrics. Already indexed rows survive VM artifacts being absent.
Source paths and hashes identify the evidence used for each result. The committed
Flat registry validation score is indexed as `REGISTRY_ONLY`; H1's three-task mean
is never presented as its four-class endpoint macro-F1. Missing provenance prevents
those rows from silently qualifying for final selection.

Lightweight future run/evaluation metadata is no longer blanket-ignored by Git.
Tensor checkpoints and per-image prediction files remain storage artifacts; keep
Azure/storage backups with the recorded hashes. The master index does not replace
those full files. Repeated indexing is idempotent and atomic per output file; an
interrupted multi-file rebuild can be rerun to restore a consistent derived view.

## Selection and limitations

Only validation endpoint macro-F1 at the selected checkpoint qualifies. Hierarchies
are ranked over all seven matched backbones after the 35-cell matrix is complete.
The top two advance when their means differ by at most 0.005; otherwise the top
hierarchy is reported as S*. No subsequent block is launched. Test and oracle
metrics are never inputs to this ranking.

Local evidence currently records two completed B0 runs and 33 pending cells.
Their canonical VM run directories/checkpoints/evaluations are absent locally.
The checkpoint registry preserves both existing H1 checkpoint generations.
Existing notebook outputs are preserved as evidence, not scraped into new canonical
results. Old runs need their original VM artifacts indexed; the rebuild does not
perform missing validation inference. Any such evaluation must be separately
requested using the existing evaluation API with `split='validation'`.

CUDA behavior, real dataset throughput, component checkpoint storage costs and
empirical H2-H4 performance require Azure verification after the policy decision.
Duration accounting is active orchestration time across resumed sessions, not a
hardware profiler. All baseline training hyperparameters and split bytes remain
unchanged; H1's auxiliary training task remains a disclosed difference.
