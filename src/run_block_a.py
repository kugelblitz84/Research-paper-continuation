"""Status by default; explicitly run one system or the complete Block-A matrix."""

import argparse

from src.experiment_catalog import SYSTEMS, expected_cells
from src.results_registry import matrix_status, rebuild
from src.run_experiment import run_experiment


def run_batch(system=None, seed=42):
    outcomes = []
    for cell in expected_cells(seed):
        if system and cell["system_type"] != system:
            continue
        try:
            result = run_experiment(
                cell["system_type"], cell["backbone"], seed, allow_training=True
            )
        except (ValueError, FileNotFoundError):
            # Scientific incompatibility aborts the batch rather than masking drift.
            raise
        except Exception as error:
            result = {"run_id": cell["run_id"], "status": "FAILED", "error": str(error)}
        outcomes.append(result)
        print(result, flush=True)
    rebuild(seed=seed)
    print("Batch summary:")
    for result in outcomes:
        print(result)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--system", choices=SYSTEMS)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.system or args.all:
        run_batch(args.system, args.seed)
    else:
        status = matrix_status(seed=args.seed)
        print(status.pivot(index="backbone", columns="system_code", values="status").to_string())
        print(status.loc[status.note != "", ["run_id", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
