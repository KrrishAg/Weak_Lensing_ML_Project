#!/usr/bin/env python
"""Import already-downloaded FAIR-Universe files into the project with MD5 checks.

Useful in Colab when the browser/Google Drive has the files already, or when a
runtime cannot download directly from Zenodo.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from download_data import FILES, md5sum

ROOT = Path(__file__).resolve().parents[1]


def copy_checked(src_dir: Path, dst_dir: Path, key: str, no_checksum: bool) -> None:
    item = FILES[key]
    src = src_dir / item["name"]
    dst = dst_dir / item["name"]
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    if not no_checksum:
        got = md5sum(src)
        if got != item["md5"]:
            raise RuntimeError(
                f"Checksum mismatch for {src.name}: expected {item['md5']}, got {got}"
            )
        print(f"source MD5 verified: {src.name} -> {got}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")
    else:
        print(f"already in place: {dst}")

    if not no_checksum:
        got_dst = md5sum(dst)
        if got_dst != item["md5"]:
            raise RuntimeError(f"Destination checksum mismatch after copy: {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--source-dir", required=True, help="folder containing downloaded .npy files")
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()

    src = Path(args.source_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    if args.sample:
        dst = ROOT / "data" / "sample"
        keys = ["mask", "sample_labels", "sample_maps"]
        next_cmd = "python scripts/inspect_data.py --config configs/sample.yaml"
    else:
        dst = ROOT / "data" / "raw"
        keys = ["mask", "full_labels", "full_maps"]
        next_cmd = "python scripts/inspect_data.py --config configs/default.yaml"

    for key in keys:
        copy_checked(src, dst, key, args.no_checksum)

    print(f"\nData imported successfully into {dst}")
    print("Next:", next_cmd)


if __name__ == "__main__":
    main()
