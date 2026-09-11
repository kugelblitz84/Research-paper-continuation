"""Explicit disk configuration and frozen baseline drift checks."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = (
    "efficientnet_b0",
    "densenet121",
    "densenet169",
    "resnet50",
    "mobilenet_v3_large",
    "efficientnet_b2",
    "efficientnet_b3",
)
CLASSES = ("non_malignant", "melanoma", "bcc", "scc")
TASK_CLASSES = (
    ("non_malignant", "malignant"),
    ("melanoma", "bcc", "scc"),
    ("Tis", "T1", "T2", "T3", "T4"),
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    experiment = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(experiment) != {"protocol", "architecture", "system_type", "experiment_id"}:
        raise ValueError(
            "Experiment must specify protocol, architecture, system_type, experiment_id only"
        )
    if experiment["architecture"] not in ARCHITECTURES or experiment["system_type"] not in (
        "flat",
        "shared_hard",
        "shared_soft",
        "dedicated_hard",
        "dedicated_soft",
    ):
        raise ValueError("Unsupported baseline system/backbone")
    protocol_path = (ROOT / experiment["protocol"]).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "configs/protocol_lock.json").read_text())
    if digest(protocol) != lock["scientific_config_sha256"]:
        raise ValueError("Frozen protocol changed; create a separately versioned future protocol")
    cfg = deepcopy(protocol)
    cfg.update(experiment)
    cfg["config_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return cfg


def validate_runtime_config(config):
    """Reject notebook mutations of the resolved scientific config before a real run."""
    if config != load_config(config["config_path"]):
        raise ValueError(
            "Resolved frozen config was modified in memory; create a separately versioned protocol"
        )
