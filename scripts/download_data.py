#!/usr/bin/env python
"""Download and checksum the official FAIR-Universe public training data."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://zenodo.org/records/20056065/files"

FILES = {
    "mask": {
        "name": "WIDE12H_bin2_2arcmin_mask.npy",
        "md5": "29b8bbb8a9ad41d3dacfaf61008a8268",
    },
    "sample_labels": {
        "name": "sampled_label.npy",
        "md5": "9a08befb4c635966cf61ced1136e7144",
    },
    "sample_maps": {
        "name": "sampled_WIDE12H_bin2_2arcmin_kappa.npy",
        "md5": "63388a13ed245208a05a4f66c8f188c1",
    },
    "full_labels": {
        "name": "label_newrealization.npy",
        "md5": "2b88e3ccf8c8e66b445ee649886586c7",
    },
    "full_maps": {
        "name": "WIDE12H_bin2_2arcmin_kappa_newrealization.npy",
        "md5": "330ccc9ced8288b1d4d9bd81e74228b0",
    },
}


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        # If the server ignores Range, restart rather than append duplicate bytes.
        if existing and r.status_code == 200:
            existing = 0
            mode = "wb"
        else:
            r.raise_for_status()
            mode = "ab" if existing else "wb"
        total = r.headers.get("content-length")
        total = (int(total) + existing) if total is not None else None
        with dest.open(mode) as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def fetch(item: dict, dest_dir: Path, verify: bool = True) -> None:
    dest = dest_dir / item["name"]
    if dest.exists() and verify and md5sum(dest) == item["md5"]:
        print(f"OK already present: {dest}")
        return
    url = f"{BASE}/{item['name']}?download=1"
    print(f"Downloading {url}")
    download_with_resume(url, dest)
    if verify:
        got = md5sum(dest)
        if got != item["md5"]:
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: expected {item['md5']}, got {got}"
            )
        print(f"MD5 verified: {got}")


def main() -> None:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true", help="download the 23 MB official sample")
    group.add_argument("--full", action="store_true", help="download the complete 6.8 GB training maps")
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()

    verify = not args.no_checksum
    if args.sample:
        dest = ROOT / "data" / "sample"
        keys = ["mask", "sample_labels", "sample_maps"]
    else:
        dest = ROOT / "data" / "raw"
        keys = ["mask", "full_labels", "full_maps"]
        print("Full maps are about 6.8 GB; make sure the runtime has enough disk space.")

    for key in keys:
        fetch(FILES[key], dest, verify=verify)

    print(f"\nData ready in {dest}")
    print("Next: python scripts/inspect_data.py --config configs/sample.yaml" if args.sample else
          "Next: python scripts/inspect_data.py --config configs/default.yaml")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted. Re-run the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
