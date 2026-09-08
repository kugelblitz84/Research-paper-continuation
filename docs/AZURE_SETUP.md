# Azure execution setup

This guide prepares the implementation for the future authorized B0 run. No VM was
provisioned and no Azure training was launched during the rebuild.

## Environment

Use Python 3.11.9 for historical continuity. The local build was validated on Windows
CPU with torch 2.13.0 and torchvision 0.28.0, matching the supplied version pins.
CUDA wheel variants and driver compatibility must be verified on the actual VM.
Do not silently substitute a different torch/torchvision pair.

For the Azure Linux GPU driver, follow Microsoft's
[N-series GPU driver setup](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup).
Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
and its previous-version instructions to choose a wheel compatible with the VM's
CUDA driver and hardware. The selector's current defaults are not the frozen
historical version requirement.

```bash
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching CUDA-enabled torch==2.13.0 / torchvision==0.28.0 wheels
# using the official compatible wheel source for this VM.
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m ipykernel install --prefix .venv --name skin-cancer --display-name "Skin cancer (Azure)"
```

Do not copy `validation/installed_packages.json` as a Linux lockfile: it records the
Windows validation environment and includes platform-specific dependencies.
`requirements.txt` preserves scientific package versions, while each run captures
its actual environment and resolved pretrained weights.

## Data and persistent storage

Mount persistent storage for raw images and experiment artifacts. Set:

```bash
export SKIN_CANCER_DATA_ROOT=/path/to/historical-dataset-root
export PYTHONHASHSEED=42
```

The root contains `data/raw/isic2019/images/ISIC_2019_Training_Input/` and
`data/raw/emb/images/isic/`. Preserve the repository's exact split manifests.
Checkpoint generations are written beneath this checkout's `models/checkpoints/`;
place the checkout on durable storage or configure a durable filesystem mount at
that directory. No automatic retention cleanup is performed.

## Preflight and execution order

1. Run `python -m pytest -q` and `python tools/validate_notebooks.py` on the VM.
2. Launch `python -m jupyterlab notebooks --no-browser --ip=127.0.0.1`; use your
   existing secure remote access workflow and select the installed Python kernel.
3. Run notebooks 00 and 01; enable image SHA-256 verification in 01.
4. After explicit training authorization, set `RUN_TRAINING=True` in notebook 02
   for `configs/experiments/flat/efficientnet_b0.yaml`. Run it to completion.
5. Run notebook 03 for `configs/experiments/shared_hard/efficientnet_b0.yaml`.
6. Freeze both selected checkpoints, back up artifacts, then run notebook 04.
7. Run notebook 05; inspect historical/new/difference, intervals and routing metrics.
8. Only after accepting the B0 reproduction review may the remaining six pairs run.

Training automatically checks raw train/validation image hashes before optimization.
ImageNet weight downloads may occur on the first real model build. Offline training
requires those exact torchvision weights to be cached first; the local build tests
used `pretrained=False` and did not download them.

## Interruption

See [CHECKPOINTS_AND_RECOVERY.md](CHECKPOINTS_AND_RECOVERY.md). Use the same package
versions/config/manifests and explicit `RESUME=True`. The latest *completed epoch*
is resumable. Multiworker augmentation streams are not promised bit-identical after
VM interruption. Record resumed runs as resumed evidence.
