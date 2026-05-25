"""Evaluate YOLOv8 detection metrics on a validation or test split."""

from __future__ import annotations

import argparse
from typing import Any

from utils import (
    dataset_yaml_from_config,
    get_nested,
    load_config,
    require_package,
    resolve_path,
    resolve_weights,
    setup_logger,
    ultralytics_device,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line evaluation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", help="Model weights to evaluate")
    parser.add_argument("--data", help="Dataset YAML path or Ultralytics dataset name")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Split")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--device", help="Device: auto, cpu, 0, 1, ...")
    return parser.parse_args()


def summarize_metrics(metrics: Any) -> dict[str, float]:
    """Extract the core detection metrics from an Ultralytics result object."""

    box = getattr(metrics, "box", metrics)
    precision = float(getattr(box, "mp", 0.0))
    recall = float(getattr(box, "mr", 0.0))
    f1 = 0.0 if precision + recall == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }


def main() -> None:
    """Run validation and write a metrics JSON summary."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("evaluate")
    ultralytics = require_package("ultralytics")

    validation = config.get("validation", {})
    device = args.device or get_nested(config, ["runtime", "device"], "auto")
    weights = resolve_weights(args.weights, config, logger)
    model = ultralytics.YOLO(weights)

    metrics = model.val(
        data=dataset_yaml_from_config(config, args.data),
        split=args.split or validation.get("split", "val"),
        imgsz=args.imgsz or validation.get("imgsz", 640),
        batch=args.batch or validation.get("batch", 16),
        conf=validation.get("conf", 0.001),
        iou=validation.get("iou", 0.6),
        device=ultralytics_device(device),
        project=str(
            resolve_path(
                get_nested(config, ["paths", "evaluation_project"], "runs/evaluation")
            )
        ),
        name=validation.get("name", "validation"),
        plots=True,
    )

    summary = summarize_metrics(metrics)
    output_path = write_json(
        summary,
        f"{get_nested(config, ['paths', 'evaluation_project'], 'runs/evaluation')}"
        "/metrics_summary.json",
    )
    logger.info("Metrics: %s", summary)
    logger.info("Saved metrics summary to %s", output_path)


if __name__ == "__main__":
    main()
