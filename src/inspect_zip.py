"""Inspect a Big Data Bowl archive without extracting its contents."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to the downloaded competition ZIP")
    args = parser.parse_args()
    if not args.archive.is_file():
        raise SystemExit(f"Archive not found: {args.archive}")

    with zipfile.ZipFile(args.archive) as archive:
        items = archive.infolist()
        files = [item for item in items if not item.is_dir()]
        compressed = sum(item.compress_size for item in files)
        expanded = sum(item.file_size for item in files)
        print(f"Archive: {args.archive}")
        print(f"Files: {len(files)}")
        print(f"Compressed payload: {human_size(compressed)}")
        print(f"Expanded payload: {human_size(expanded)}")
        print("\nLargest files (uncompressed size):")
        for item in sorted(files, key=lambda f: f.file_size, reverse=True)[:25]:
            print(f"{human_size(item.file_size):>10}  {item.filename}")
        print("\nLikely starter files:")
        keywords = ("players.csv", "combine_results.csv", "combine_tracking.csv",
                    "player_career_successes.csv", "player_play.csv", "games.csv")
        for item in files:
            if any(item.filename.endswith(name) for name in keywords):
                print(f"{human_size(item.file_size):>10}  {item.filename}")


if __name__ == "__main__":
    main()
