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


EXTENSION_SYSTEMS = ("shared_soft", "dedicated_hard", "dedicated_soft")
EXTENSION_PATH = "configs/block_a_extension.yaml"
EXTENSION_LOCK_PATH = "configs/block_a_extension_lock.json"


def load_extension(baseline_hash):
    """Load only the explicitly approved, locked two-task Block-A extension."""
    try:
        extension = yaml.safe_load((ROOT / EXTENSION_PATH).read_text(encoding="utf-8"))
        lock = json.loads((ROOT / EXTENSION_LOCK_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError("Approved Block-A extension or lock missing/invalid") from exc
    if not isinstance(extension, dict) or not isinstance(lock, dict):
        raise ValueError("Invalid Block-A extension/lock")
    extension_hash = digest(extension)
    if extension_hash != lock.get("scientific_config_sha256"):
        raise ValueError("Block-A extension hash mismatch")
    if (extension.get("extension_version") != "block_a_two_task_v1"
            or lock.get("extension_version") != extension["extension_version"]
            or extension.get("status") != "approved"
            or extension.get("systems") != list(EXTENSION_SYSTEMS)
            or extension.get("selection") != dict.fromkeys(EXTENSION_SYSTEMS, "validation_macro_f1")
            or extension.get("baseline_protocol", {}).get("scientific_config_sha256") != baseline_hash):
        raise ValueError("Unsupported or incompatible Block-A extension")
    return extension, extension_hash


def baseline_protocol(config):
    """Project extension-backed configs to the unchanged baseline for provenance/gates."""
    protocol = deepcopy({k: v for k, v in config.items() if k not in (
        "architecture", "system_type", "experiment_id", "config_path", "protocol",
        "block_a_extension", "extension_version", "extension_hash", "extension_path",
    )})
    if "block_a_extension" in config:
        for system in EXTENSION_SYSTEMS:
            protocol["selection"].pop(system, None)
    return protocol


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
    if experiment["system_type"] in EXTENSION_SYSTEMS:
        extension, extension_hash = load_extension(digest(protocol))
        cfg.update(block_a_extension=extension,
                   extension_version=extension["extension_version"],
                   extension_hash=extension_hash, extension_path=EXTENSION_PATH)
        cfg["selection"].update(extension["selection"])
    cfg.update(experiment)
    cfg["config_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return cfg


def validate_runtime_config(config):
    """Reject notebook mutations of the resolved scientific config before a real run."""
    if config != load_config(config["config_path"]):
        raise ValueError(
            "Resolved frozen config was modified in memory; create a separately versioned protocol"
        )
