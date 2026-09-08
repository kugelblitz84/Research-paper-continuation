"""Restore optional historical parity evidence from the user-supplied ZIP."""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    inventory = json.loads((ROOT / "docs/source_inventory.json").read_text())
    if hashlib.sha256(args.zip_path.read_bytes()).hexdigest() != inventory["zip_sha256"]:
        raise ValueError("This ZIP differs from the audited historical source")
    target = ROOT / "reference/historical"
    with zipfile.ZipFile(args.zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            raw = PurePosixPath(info.filename)
            if raw.is_absolute() or ".." in raw.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe archive member")
            relative = raw.relative_to("research_paper")
            if relative.parts[0] not in ("src", "scripts", "configs", "reports", "tests") and str(
                relative
            ) not in ("README.md", "requirements.txt"):
                continue
            output = target.joinpath(*relative.parts).resolve()
            if not output.is_relative_to(target.resolve()):
                raise ValueError("Archive path escape")
            data = archive.read(info)
            if output.exists():
                if output.read_bytes() != data:
                    raise FileExistsError(f"Changed reference: {output}")
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as f:
                f.write(data)
    print("Historical reference restored; new src never imports it")


if __name__ == "__main__":
    main()
