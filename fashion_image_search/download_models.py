"""Pre-download Hugging Face models used by the real backend."""

from __future__ import annotations

import argparse

from fashion_image_search.common.config import MODEL
from fashion_image_search.indexer.detect import HuggingFaceFashionDetector
from fashion_image_search.indexer.embed import HuggingFaceFashionEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache Hugging Face models.")
    parser.add_argument("--device", default="cpu", help="Use cpu for download-only warmup.")
    args = parser.parse_args()

    print(f"Downloading detector: {MODEL.detector_name}")
    HuggingFaceFashionDetector(model_name=MODEL.detector_name, device=args.device)
    print(f"Downloading encoder: {MODEL.encoder_name}")
    HuggingFaceFashionEncoder(model_name=MODEL.encoder_name, device=args.device)
    print("Models are cached and ready.")


if __name__ == "__main__":
    main()
