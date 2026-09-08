"""Byte-locked manifests and split-specific image datasets; no split generation."""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.config import ROOT, TASK_CLASSES
from src.reproducibility import seed_worker, sha256
from src.transforms import build_transform

DIAGNOSES = {
    "melanocytic_nevus": 0,
    "benign_keratosis_like_lesion": 0,
    "dermatofibroma": 0,
    "vascular_lesion": 0,
    "melanoma": 1,
    "basal_cell_carcinoma": 2,
    "squamous_cell_carcinoma": 3,
}


def resolve_data_root(data_root=None):
    raw = data_root or os.environ.get("SKIN_CANCER_DATA_ROOT")
    if not raw:
        raise ValueError(
            "Set SKIN_CANCER_DATA_ROOT to the directory containing data/raw/ from the historical layout"
        )
    return Path(raw).expanduser().resolve()


def read_manifest(config, source):
    path = ROOT / config["manifests"][source]
    lock = json.loads((ROOT / "data/manifests/manifest_lock.json").read_text())
    if sha256(path) != lock[path.name]:
        raise ValueError(f"Frozen manifest hash mismatch: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if source == "isic":
        frame = frame.loc[(frame.split_included == "1") & (frame.include_stage_1 == "1")].copy()
        frame["target"] = frame.diagnosis_canonical.str.strip().map(DIAGNOSES)
        expected_dataset = "isic2019"
    else:
        frame = frame.loc[frame.modality.str.strip().str.lower() == "dermoscopic"].copy()
        frame["target"] = frame.t_category.str.strip().map(dict(zip(TASK_CLASSES[2], range(5))))
        if not (frame.derived_stage_ajcc.astype(float) == frame.target).all():
            raise ValueError("T-category/derived-stage disagreement")
        expected_dataset = "isic_stage03"
    if set(frame.dataset) != {expected_dataset} or frame.target.isna().any():
        raise ValueError("Unexpected dataset or labels")
    if frame.image_id.duplicated().any() or not set(frame.split) <= {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("Duplicate IDs or invalid split")
    frame["target"] = frame.target.astype(int)
    return frame.reset_index(drop=True)


def audit_manifests(config):
    frames = {s: read_manifest(config, s) for s in ("isic", "stage3")}
    result = {}
    for source, f in frames.items():
        for key in (
            "image_id",
            "split_group_id",
            "file_sha256",
            "patient_id",
            "lesion_id",
        ):
            if key in f:
                nonempty = f.loc[f[key] != ""]
                if (nonempty.groupby(key).split.nunique() > 1).any():
                    raise ValueError(f"Cross-split {source}/{key} leakage")
        counts = f.groupby("split").size().to_dict()
        if counts != config["expected_counts"][source]:
            raise ValueError(f"Unexpected {source} cohort counts: {counts}")
        result[source] = {
            "counts": counts,
            "class_counts": {
                s: f.loc[f.split == s].target.value_counts().sort_index().to_dict() for s in counts
            },
            "sha256": sha256(ROOT / config["manifests"][source]),
        }
    # Report source overlap; never silently alter historical memberships.
    cross = []
    for key in ("image_id", "file_sha256", "patient_id", "lesion_id"):
        a, b = frames["isic"], frames["stage3"]
        pairs = a.loc[a[key] != "", [key, "split"]].merge(
            b.loc[b[key] != "", [key, "split"]], on=key, suffixes=("_isic", "_stage3")
        )
        bad = pairs.loc[pairs.split_isic != pairs.split_stage3]
        if len(bad):
            cross.append(
                {
                    "key": key,
                    "count": len(bad),
                    "examples": bad.head(5).to_dict("records"),
                }
            )
    result["cross_source_split_conflicts"] = cross
    return result


def encode_targets(endpoint, source="isic"):
    if source == "stage3":
        if endpoint not in range(5):
            raise ValueError("Bad T-category")
        return torch.tensor([-100, -100, endpoint]), torch.tensor([False, False, True])
    if endpoint not in range(4):
        raise ValueError("Bad endpoint")
    return torch.tensor(
        [int(endpoint > 0), endpoint - 1 if endpoint else -100, -100]
    ), torch.tensor([True, endpoint > 0, False])


class SkinDataset(Dataset):
    def __init__(self, config, source, split, data_root, *, train=False, verify=False):
        if split not in ("train", "validation", "test") or (train and split != "train"):
            raise ValueError("Invalid split/augmentation combination")
        frame = read_manifest(config, source)
        self.frame = frame.loc[frame.split == split].reset_index(drop=True)
        self.split, self.source = split, source
        self.transform = build_transform(config, train)
        self.root = resolve_data_root(data_root)
        self.paths = []
        for raw in self.frame.image_path:
            path = (self.root / raw).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Image path escapes configured root")
            self.paths.append(path)
        if verify:
            missing = [str(p) for p in self.paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} missing {source}/{split} images; first: {missing[0]}"
                )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.paths[index]) as opened:
            image = self.transform(opened.convert("RGB"))
        target = int(row.target)
        targets, masks = encode_targets(target, self.source)
        return dict(
            image=image,
            target=torch.tensor(target),
            targets=targets,
            masks=masks,
            image_id=row.image_id,
        )


def make_loader(dataset, config, *, train=False, workers=None):
    p = config["loader"]
    count = p["workers"] if workers is None else workers
    args = dict(
        batch_size=p["batch_size"],
        shuffle=train,
        num_workers=count,
        pin_memory=p["pin_memory"],
        persistent_workers=p["persistent_workers"] and count > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    if count > 0:
        args["prefetch_factor"] = p["prefetch_factor"]
    return DataLoader(dataset, **args)


def training_loaders(config, data_root, *, workers=None):
    """Only train and validation datasets can be constructed through this API."""
    a = SkinDataset(config, "isic", "train", data_root, train=True, verify=True)
    v = SkinDataset(config, "isic", "validation", data_root, verify=True)
    if config["system_type"] == "flat":
        datasets = {"train": a, "validation": v}
    else:
        b = SkinDataset(config, "stage3", "train", data_root, train=True, verify=True)
        w = SkinDataset(config, "stage3", "validation", data_root, verify=True)
        datasets = {
            "train": ConcatDataset([a, b]),
            "task1": v,
            "task2": Subset(v, v.frame.index[v.frame.target > 0].tolist()),
            "task3": w,
        }
    return {
        k: make_loader(d, config, train=k == "train", workers=workers) for k, d in datasets.items()
    }


def verify_images(config, data_root, *, splits=("train", "validation"), hashes=True):
    report = {}
    sources = ("isic", "stage3") if config["system_type"] == "shared_hard" else ("isic",)
    for source in sources:
        for split in splits:
            d = SkinDataset(config, source, split, data_root, verify=True)
            for p, expected in zip(d.paths, d.frame.file_sha256):
                if hashes and sha256(p) != expected:
                    raise ValueError(f"Image SHA256 mismatch: {p}")
            report[f"{source}/{split}"] = {"images": len(d), "hashes_verified": hashes}
    return report
