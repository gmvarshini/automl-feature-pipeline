"""Command line entry point for the training pipeline.

Running ``python -m src.main train`` executes feature engineering and AutoML
training end to end and writes the model and metadata to disk. Serving is kept
deliberately separate and is launched through uvicorn against
``src.serve:app`` rather than from this script.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import get_config
from src.training import train_with_automl


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the requested command.

    Args:
        argv: Optional argument list (defaults to ``sys.argv`` when ``None``).

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description="AutoML feature pipeline.")
    parser.add_argument(
        "command",
        choices=["train"],
        help="'train' runs feature engineering and AutoML training end to end.",
    )
    args = parser.parse_args(argv)

    if args.command == "train":
        config = get_config()
        print(
            f"Training on target '{config.target_column}' "
            f"({config.task_type}) with a "
            f"{config.automl_time_budget_seconds}s budget..."
        )
        metadata = train_with_automl(config)
        print("Training complete. Selected model and performance:")
        print(json.dumps(metadata["test_metrics"], indent=2))
        print(f"Selected model: {metadata['selected_model']}")
        print(f"Model written to: {config.model_output_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
