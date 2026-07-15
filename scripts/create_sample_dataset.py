"""Create a deterministic random image sample for indexing experiments."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def create_sample(source: Path, destination: Path, count: int, seed: int) -> int:
    source = source.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if count > len(images):
        raise ValueError(f"Requested {count} images, but only found {len(images)} in {source}")
    random.Random(seed).shuffle(images)
    selected = images[:count]
    for path in selected:
        shutil.copy2(path, destination / path.name)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a deterministic random image sample.")
    parser.add_argument("--source", type=Path, default=Path("val_test2020/test"))
    parser.add_argument("--destination", type=Path, default=Path("val_test2020_sample_1600/test"))
    parser.add_argument("--count", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    copied = create_sample(args.source, args.destination, args.count, args.seed)
    print(f"Copied {copied} images to {args.destination}")


if __name__ == "__main__":
    main()
