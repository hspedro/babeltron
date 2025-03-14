#!/usr/bin/env python3
"""
Script to download translation models for Babeltron.
"""
import argparse
import os
from pathlib import Path
from typing import Optional, Union

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

# Model configurations
MODEL_CONFIGS = {
    "m2m100": {
        "sizes": os.environ.get("BABELTRON_M2M_SIZES", "418M,1.2B,12B").split(","),
        "default_size": os.environ.get("BABELTRON_DEFAULT_M2M_SIZE", "418M"),
        "model_name_format": "facebook/m2m100_{size}",
        "dir_name_format": "m2m100-{size}",
        "tokenizer_class": M2M100Tokenizer,
        "model_class": M2M100ForConditionalGeneration,
    },
    "nllb": {
        "sizes": os.environ.get("BABELTRON_NLLB_SIZES", "600M,3.3B").split(","),
        "default_size": os.environ.get("BABELTRON_DEFAULT_NLLB_SIZE", "600M"),
        "model_name_format": "facebook/nllb-200-{size_prefix}{size}",
        "dir_name_format": "nllb-{size}",
        "tokenizer_class": AutoTokenizer,
        "model_class": AutoModelForSeq2SeqLM,
    },
}

DEFAULT_MODEL_TYPE = os.environ.get("BABELTRON_DEFAULT_MODEL_TYPE", "m2m100")
DEFAULT_OUTPUT_DIR: Path = Path.home() / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Download translation models")
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_CONFIGS.keys()),
        default=DEFAULT_MODEL_TYPE,
        help=f"Model type to download (default: {DEFAULT_MODEL_TYPE})",
    )

    # Create help text that shows available sizes for each model type
    size_help = "Model size to download. Available sizes by model type: "
    size_help += " | ".join(
        [
            f"{model_type}: {', '.join(config['sizes'])} (default: {config['default_size']})"
            for model_type, config in MODEL_CONFIGS.items()
        ]
    )

    parser.add_argument(
        "--size",
        help=size_help,
    )

    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save the model"
    )

    args = parser.parse_args()

    # Validate and set default size based on model type
    model_config = MODEL_CONFIGS[args.model_type]
    valid_sizes = model_config["sizes"]

    if args.size is None:
        args.size = model_config["default_size"]
    elif args.size not in valid_sizes:
        parser.error(
            f"For model type '{args.model_type}', size must be one of {valid_sizes}"
        )

    return args


def download_model(
    model_type: str = DEFAULT_MODEL_TYPE,
    model_size: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Download translation model and tokenizer.

    Args:
        model_type (str): Type of model to download ('m2m100' or 'nllb')
        model_size (str, optional): Size of the model to download (depends on model type)
        output_dir (str or Path, optional): Directory to save the model to

    Returns:
        str: Path to the downloaded model directory
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type must be one of {list(MODEL_CONFIGS.keys())}")

    model_config = MODEL_CONFIGS[model_type]

    if model_size is None:
        model_size = model_config["default_size"]
    elif model_size not in model_config["sizes"]:
        raise ValueError(
            f"For model type '{model_type}', size must be one of {model_config['sizes']}"
        )

    # Get the HuggingFace model name
    if model_type == "nllb":
        # For NLLB models, use different prefixes based on the model size
        if model_size == "600M":
            size_prefix = "distilled-"
        else:
            size_prefix = ""
        model_name = model_config["model_name_format"].format(
            size=model_size, size_prefix=size_prefix
        )
    else:
        # For other model types (e.g., m2m100)
        model_name = model_config["model_name_format"].format(size=model_size)

    # Get the directory name for the model
    dir_name = model_config["dir_name_format"].format(size=model_size)

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a subdirectory for the specific model
    model_dir = output_dir / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # For backward compatibility, create a symlink with the old naming convention
    old_style_dir = output_dir / f"{model_type}_{model_size}"
    if not old_style_dir.exists():
        try:
            # Create relative symlink
            old_style_dir.symlink_to(dir_name, target_is_directory=True)
            print(
                f"Created backward compatibility symlink: {old_style_dir} -> {dir_name}"
            )
        except Exception as e:
            print(f"Warning: Could not create backward compatibility symlink: {e}")

    print(f"Downloading {model_name} model and tokenizer to {model_dir}...")

    print("Downloading tokenizer...")
    tokenizer_class = model_config["tokenizer_class"]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)

    print("Downloading model (this may take a while)...")
    model_class = model_config["model_class"]
    model = model_class.from_pretrained(model_name)
    model.save_pretrained(model_dir)

    print(f"Model and tokenizer successfully saved to {model_dir}")

    return str(model_dir)


def main():
    args = parse_args()

    try:
        print(f"Downloading {args.model_type} model (size: {args.size})...")
        print(
            "This may take a while depending on your internet connection and the model size."
        )

        output_path = download_model(
            model_type=args.model_type, model_size=args.size, output_dir=args.output_dir
        )

        print(f"Model successfully downloaded and saved to {output_path}")

    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
