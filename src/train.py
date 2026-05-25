"""Train or fine-tune a YOLOv8 object detection model with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    copy_training_weights,
    dataset_yaml_from_config,
    get_nested,
    load_config,
    require_package,
    resolve_path,
    setup_logger,
    ultralytics_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", help="Pretrained or custom YOLO weights")
    parser.add_argument("--data", help="Dataset YAML path or Ultralytics dataset name")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--device", help="Device: auto, cpu, 0, 1, ...")
    parser.add_argument("--name", help="Ultralytics run name")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()


def build_train_kwargs(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build Ultralytics ``YOLO.train`` keyword arguments."""

    training = config.get("training", {})
    augmentation = config.get("augmentation", {}).get("builtin", {})
    device = args.device or get_nested(config, ["runtime", "device"], "auto")

    kwargs: dict[str, Any] = {
        "data": dataset_yaml_from_config(config, args.data),
        "epochs": args.epochs or training.get("epochs", 50),
        "imgsz": args.imgsz or training.get("imgsz", 640),
        "batch": args.batch or training.get("batch", 16),
        "device": ultralytics_device(device),
        "workers": training.get("workers", 4),
        "patience": training.get("patience", 25),
        "cache": training.get("cache", False),
        "project": str(
            resolve_path(get_nested(config, ["paths", "train_project"], "runs/train"))
        ),
        "name": args.name or training.get("name", "yolov8_custom"),
        "exist_ok": training.get("exist_ok", True),
        "optimizer": training.get("optimizer", "auto"),
        "seed": training.get("seed", get_nested(config, ["project", "seed"], 42)),
        "amp": training.get("amp", True),
        "close_mosaic": training.get("close_mosaic", 10),
        "resume": args.resume,
        "plots": True,
    }

    if config.get("augmentation", {}).get("use_ultralytics_builtin", True):
        # These names match Ultralytics' training API.
        kwargs.update(augmentation)

    return kwargs


def main() -> None:
    """Train the configured YOLOv8 model and copy weights to ``models/``."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("train")
    ultralytics = require_package("ultralytics")

    model_name = args.model or get_nested(config, ["model", "pretrained"], "yolov8n.pt")
    model = ultralytics.YOLO(model_name)
    train_kwargs = build_train_kwargs(config, args)

    logger.info("Starting training with model=%s", model_name)
    logger.info("Training arguments: %s", train_kwargs)
    model.train(**train_kwargs)

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", train_kwargs["project"]))
    copied = copy_training_weights(
        save_dir,
        get_nested(config, ["paths", "models_dir"], "models"),
    )
    logger.info("Training complete. Copied weights: %s", [str(path) for path in copied])


if __name__ == "__main__":
    main()
