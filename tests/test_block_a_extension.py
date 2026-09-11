
"""Contracts for the approved Block-A H2/H3/H4 protocol extension."""

import json
from copy import deepcopy

import pytest
import yaml

import src.config as config_module
from src.config import (
    EXTENSION_SYSTEMS,
    baseline_protocol,
    digest,
    load_config,
    load_extension,
)


EXTENSION_VERSION = "block_a_two_task_v1"

EXTENSION_HASH = (
    "2126d125ae9b0776a652ee55cdef8aa99a3e9ad6f46ed04ae9526203fc8c8b57"
)

BASELINE_PROTOCOL_HASH = (
    "1b4a63c8dda7956f0755f8a132e1d7042945a1761d55e6b3e81d9cf58868e796"
)

FLAT_B0_HASH = (
    "e9bb47a6117aaa585a978e849d78d8c98df340643ccc06ba688f14cfa3da92c0"
)

H1_B0_HASH = (
    "80870a870afb648bef498a6baf95a3a53198eefd160085c9731c65407b01f3c1"
)


def test_extension_identity_and_lock():
    extension, extension_hash = load_extension(BASELINE_PROTOCOL_HASH)

    assert extension["extension_version"] == EXTENSION_VERSION
    assert extension["status"] == "approved"
    assert tuple(extension["systems"]) == EXTENSION_SYSTEMS

    assert extension_hash == EXTENSION_HASH
    assert digest(extension) == EXTENSION_HASH


def test_historical_flat_and_h1_hashes_unchanged():
    flat = load_config(
        "configs/experiments/flat/efficientnet_b0.yaml"
    )
    h1 = load_config(
        "configs/experiments/shared_hard/efficientnet_b0.yaml"
    )

    assert digest(flat) == FLAT_B0_HASH
    assert digest(h1) == H1_B0_HASH

    assert "block_a_extension" not in flat
    assert "block_a_extension" not in h1
    assert "extension_hash" not in flat
    assert "extension_hash" not in h1


@pytest.mark.parametrize(
    "system",
    [
        "shared_soft",
        "dedicated_hard",
        "dedicated_soft",
    ],
)
def test_extension_backed_system_selection(system):
    config = load_config(
        f"configs/experiments/{system}/efficientnet_b0.yaml"
    )

    assert config["selection"][system] == "validation_macro_f1"
    assert config["extension_version"] == EXTENSION_VERSION
    assert config["extension_hash"] == EXTENSION_HASH

    assert (
        digest(config["block_a_extension"])
        == EXTENSION_HASH
    )


def test_h1_historical_selection_remains_unchanged():
    config = load_config(
        "configs/experiments/shared_hard/efficientnet_b0.yaml"
    )

    assert (
        config["selection"]["shared_hard"]
        == "mean_task1_task2_task3_macro_f1"
    )


@pytest.mark.parametrize(
    "system",
    [
        "shared_soft",
        "dedicated_hard",
        "dedicated_soft",
    ],
)
def test_extension_projects_back_to_exact_baseline_protocol(system):
    extended = load_config(
        f"configs/experiments/{system}/efficientnet_b0.yaml"
    )

    projected = baseline_protocol(extended)

    baseline = yaml.safe_load(
        (
            config_module.ROOT
            / "configs/protocol.yaml"
        ).read_text(encoding="utf-8")
    )

    assert projected == baseline
    assert digest(projected) == BASELINE_PROTOCOL_HASH


def test_extension_has_no_task3_or_threshold_tuning():
    extension, _ = load_extension(BASELINE_PROTOCOL_HASH)

    assert extension["tasks"]["task3"] == "absent"
    assert extension["routing"]["threshold_tuning"] is False
    assert extension["routing"]["confidence_threshold"] is None


def test_extension_explicitly_excludes_test_selection():
    extension, _ = load_extension(BASELINE_PROTOCOL_HASH)

    assert (
        extension["selection_definition"]["split"]
        == "validation"
    )

    assert (
        extension["selection_definition"]["test_data"]
        == "excluded from training and checkpoint selection"
    )


def test_extension_tampering_is_rejected(tmp_path, monkeypatch):
    source_extension = (
        config_module.ROOT
        / "configs/block_a_extension.yaml"
    )

    extension = yaml.safe_load(
        source_extension.read_text(encoding="utf-8")
    )

    tampered = deepcopy(extension)
    tampered["selection"]["shared_soft"] = (
        "mean_task1_task2_macro_f1"
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    (
        config_dir / "block_a_extension.yaml"
    ).write_text(
        yaml.safe_dump(tampered, sort_keys=False),
        encoding="utf-8",
    )

    (
        config_dir / "block_a_extension_lock.json"
    ).write_text(
        json.dumps(
            {
                "extension_version": EXTENSION_VERSION,
                "scientific_config_sha256": EXTENSION_HASH,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        config_module,
        "ROOT",
        tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="hash mismatch",
    ):
        config_module.load_extension(
            BASELINE_PROTOCOL_HASH
        )