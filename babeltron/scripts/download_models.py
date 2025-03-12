#!/usr/bin/env python3
"""
Script to download M2M-100 models for Babeltron.
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional, Union

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

VALID_MODEL_SIZES: List[str] = os.environ.get(
    "BABELTRON_MODEL_SIZES", "418M,1.2B,12B"
).split(",")
DEFAULT_MODEL_SIZE: str = os.environ.get("BABELTRON_DEFAULT_MODEL_SIZE", "418M")
DEFAULT_OUTPUT_DIR: Path = Path.home() / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Download M2M100 translation models")
    parser.add_argument(
        "--size",
        choices=VALID_MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="Model size to download (418M, 1.2B, or 12B)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save the model"
    )
    return parser.parse_args()


def download_model(
    model_size: str = DEFAULT_MODEL_SIZE,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Download M2M-100 model and tokenizer.

    Args:
        model_size (str): Size of the model to download (418M, 1.2B, or 12B)
        output_dir (str or Path, optional): Directory to save the model to

    Returns:
        str: Path to the downloaded model directory
    """
    if model_size not in VALID_MODEL_SIZES:
        raise ValueError(f"Model size must be one of {VALID_MODEL_SIZES}")

    model_name = f"facebook/m2m100_{model_size}"

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_name} model and tokenizer to {output_dir}...")

    print("Downloading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print("Downloading model (this may take a while)...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    print(f"Model and tokenizer successfully saved to {output_dir}")

    return str(output_dir)


def main():
    args = parse_args()

    try:
        model_map = {"418M": "418M", "1.2B": "1.2B", "12B": "12B"}

        model_size = model_map[args.size]
        output_dir = args.output_dir

        print(f"Downloading {args.size} model...")
        print(
            "This may take a while depending on your internet connection and the model size."
        )

        download_model(model_size=model_size, output_dir=output_dir)

        print(f"Model successfully downloaded and saved to {output_dir}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
