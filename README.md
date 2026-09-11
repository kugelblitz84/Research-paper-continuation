# Skin cancer research continuation

An organized notebook-first reconstruction of the historical Flat versus Shared-Hard
experiment. Reusable implementations live in `src/`; notebooks configure, call,
display and save. This is a clean rebuild, not a runtime wrapper around the old repo.

**Current state:** the registry preserves completed EfficientNet-B0 Flat and Shared-Hard
runs. See [Block-A infrastructure](docs/BLOCK_A_INFRASTRUCTURE.md) for the five-system,
seven-backbone matrix, result indexing, and safe execution interfaces. H2-H4 training
awaits the [explicit selection-policy decision](docs/BLOCK_A_PROTOCOL_PROPOSAL.md).
The original other-backbone reproduction-review gate remains in force.

## Start here

Open [00_environment_and_protocol_audit.ipynb](notebooks/00_environment_and_protocol_audit.ipynb).
Follow notebooks 00 through 05 in order. Every notebook runs in a fresh kernel with
safe defaults. Missing images/checkpoints are shown as not configured/not run.

On this Windows workspace, the isolated Python environment is already installed:

```powershell
.venv/Scripts/python.exe -m jupyterlab notebooks
.venv/Scripts/python.exe -m pytest -q
```

For a fresh environment with `uv`:

```text
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
uv pip install --python .venv/Scripts/python.exe -e . --no-deps
```

Linux uses `.venv/bin/python` instead. See [Azure setup](docs/AZURE_SETUP.md) for
GPU-specific checks. Tested runtime: Python 3.11.9, PyTorch 2.13.0+cpu,
torchvision 0.28.0+cpu. CPU validation is not CUDA validation.

## What is preserved

- Seven backbones: `efficientnet_b0`, `densenet121`, `densenet169`, `resnet50`,
  `mobilenet_v3_large`, `efficientnet_b2`, `efficientnet_b3`.
- Flat endpoint order: `non_malignant`, `melanoma`, `bcc`, `scc`.
- Exact historical manifest bytes and eligibility, 224-pixel transforms, seed 42,
  ImageNet initialization, losses, AdamW/cosine schedule and validation selection.
- One shared encoder. The original Shared-Hard model trains **three** task heads
  (2/3/5 logits), including the auxiliary T-category head. Only Stage 1/2 determine
  the four-class hard-routing endpoint. Removing Task3 changes the baseline.
- Historical Flat MobileNetV3 retains its 960-to-1280 Linear/Hardswish projection.
  It is not silently replaced by a universal shallow head.

The exact source-to-implementation map is in
[HISTORICAL_PROTOCOL_AUDIT.md](docs/HISTORICAL_PROTOCOL_AUDIT.md).
[Unresolved items](docs/UNRESOLVED_PROTOCOL_ITEMS.md) distinguish missing evidence
from implemented behavior. The new H2-H4 architecture and orchestration additions are described in the Block-A documentation; their scientific extension remains pending.

## Repository layout

```text
configs/protocol.yaml                 frozen scientific parameters
configs/protocol_lock.json            scientific configuration digest
configs/experiments/flat/             seven architecture selections
configs/experiments/shared_hard/      seven architecture selections
data/manifests/                      original CSV bytes + SHA-256 locks
src/models/                          central encoder factory and five systems
src/                                 data, transforms, losses, training, evaluation,
                                     routing, metrics, statistics, provenance, reporting
notebooks/00_... through 05_...       researcher workflow
tests/                               CPU unit and synthetic integration tests
tools/                               notebook QA and historical evidence replay
experiments/experiment_registry.csv   append-only run-state events
experiments/runs/<id>/                config/environment/history/logs/checkpoint refs
experiments/evaluations/<id>_<split>/ frozen evaluation metadata and metrics
models/checkpoints/<id>/epoch_.../    immutable best.pt and last.pt generations
results/historical/                  original evidence, explicitly labelled
results/predictions/                 reconstructed per-image CSVs
results/tables/                      reconstructed comparisons and bootstrap samples
results/figures/                     reconstructed figures
docs/                                protocol, setup, handover and continuation log
validation/                          actual build verification records
reference/historical/                ignored ZIP evidence; never imported by src
```

## Dataset setup

Set `SKIN_CANCER_DATA_ROOT` to a directory containing the historical `data/raw/`
layout. The manifests themselves remain unchanged in this repository.

```text
<SKIN_CANCER_DATA_ROOT>/
  data/raw/isic2019/images/ISIC_2019_Training_Input/ISIC_0000000.jpg
  data/raw/emb/images/isic/ISIC_0020963.jpg
```

The `emb` path is a legacy directory name: the frozen auxiliary cohort is the
**ISIC-derived melanoma T-category subset**. Do not substitute a different EMB
cohort or regenerate its split.

```powershell
$env:SKIN_CANCER_DATA_ROOT = 'D:/skin-cancer-dataset'
```

Notebook 01 verifies manifests immediately. Set `VERIFY_IMAGES=True` after image
setup to check train/validation file existence and SHA-256. Real training repeats
those checks. Notebook 04 separately verifies the frozen endpoint test images.
Raw images are not included, downloaded, or committed by this rebuild.

## Config and training controls

Each experiment YAML selects a system and backbone and points to the shared frozen
protocol. Scientific parameters are centralized and digest-checked; changing a
notebook's in-memory config is rejected before real training. Future research must
use a separately versioned protocol rather than modifying a completed baseline.

Notebooks 02/03 have `RUN_TRAINING=False` by default. After explicit training
approval, enable it for B0 only. `RESUME=True` resumes an interrupted run from the
latest recorded immutable `last.pt`; it refuses completed runs and changed configs
or manifests. See [checkpoint recovery](docs/CHECKPOINTS_AND_RECOVERY.md).

Notebook 04 evaluates frozen completed runs only. Notebook 05 pairs exact image IDs
and truth, saves all 10,000 bootstrap replicates, runs exact McNemar, and displays
historical/reconstructed/difference tables. The consumed internal test must never
be used to choose hyperparameters or checkpoints.

The final review control in notebook 05 releases the remaining six pairs only after
explicit researcher acceptance with a name, rationale, and hashed B0 evidence.
There is no invented numerical acceptance threshold or automatic all-backbone loop.

## Checkpoint preservation

**Git does not preserve ignored checkpoint files.** `.pt` files, raw images and run
outputs are ignored. They are expensive research artifacts, not disposable cache.
Keep a separate durable backup of `models/checkpoints/`, `experiments/runs/`,
`experiments/evaluations/`, `results/predictions/` and all completed result tables.
No checkpoint file is overwritten by the training code: each epoch uses a new
immutable directory, and `checkpoints.json` identifies the latest/best generation.
This deliberate retention uses more disk space than keeping two mutable files.

## Verification commands

```text
python -m pytest -q --junitxml=validation/pytest.xml
python -m ruff check src tests tools
python -m ruff format --check src tests tools
python tools/validate_notebooks.py
python tools/replay_historical_statistics.py
python tools/verify_historical_parity.py
```

Restore the optional reference with `python tools/import_reference.py /path/to/research_paper.zip`.
The last command additionally requires the ignored historical reference extracted
from the supplied ZIP. All normal notebooks, training, tests and stored-statistics
replay run without importing or depending on that reference tree.

See [BUILD_HANDOVER.md](docs/BUILD_HANDOVER.md) for actual results and exact next action.
