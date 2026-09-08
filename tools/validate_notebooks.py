"""Validate schema and execute each notebook in a fresh kernel; never enable training."""

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "validation/executed"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
        client = NotebookClient(
            nb,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.execute()
        nbformat.write(nb, output / path.name)
        rows.append(
            dict(
                notebook=path.name,
                schema="PASS",
                fresh_kernel_execution="PASS",
                code_cells=sum(c.cell_type == "code" for c in nb.cells),
                mode="default safe controls; real training/evaluation not executed",
            )
        )
        print(path.name, "PASS", flush=True)
    (ROOT / "validation/notebook_validation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
