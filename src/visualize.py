"""Visualization helpers for YOLO datasets, predictions, and training runs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from utils import image_files, load_config, resolve_path, setup_logger
from preprocess import load_dataset_names, parse_yolo_label


def parse_args() -> argparse.Namespace:
    """Parse visualization command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--plot",
        choices=["classes", "training"],
        default="classes",
        help="Plot type to generate.",
    )
    parser.add_argument("--results-csv", help="Ultralytics results.csv path")
    return parser.parse_args()


def yolo_to_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized YOLO ``xywh`` coordinates to pixel ``xyxy``."""

    x_center, y_center, box_width, box_height = bbox
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)
    return x1, y1, x2, y2


def draw_detections(
    image: Any,
    detections: list[dict[str, Any]],
    names: dict[int, str] | None = None,
) -> Any:
    """Draw bounding boxes and confidence scores on an OpenCV image."""

    cv2 = __import__("cv2")
    output = image.copy()
    class_names = names or {}

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["bbox_xyxy"]]
        class_id = int(detection["class_id"])
        confidence = detection.get("confidence")
        label = class_names.get(class_id, str(class_id))
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (34, 197, 94), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (34, 197, 94),
            2,
            cv2.LINE_AA,
        )

    return output


def plot_class_distribution(dataset_yaml: str | Path, output_path: str | Path) -> Path:
    """Plot the number of labels per class in the training split."""

    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot", fromlist=["pyplot"])

    names = load_dataset_names(dataset_yaml)
    counts: Counter[int] = Counter()
    for label_path in resolve_path("data/labels/train").glob("*.txt"):
        counts.update(box[0] for box in parse_yolo_label(label_path))

    labels = [names.get(class_id, str(class_id)) for class_id in sorted(counts)]
    values = [counts[class_id] for class_id in sorted(counts)]

    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pyplot.figure(figsize=(10, 5))
    pyplot.bar(labels, values, color="#2563eb")
    pyplot.xlabel("Class")
    pyplot.ylabel("Bounding boxes")
    pyplot.title("Training Class Distribution")
    pyplot.xticks(rotation=30, ha="right")
    pyplot.tight_layout()
    pyplot.savefig(output, dpi=160)
    pyplot.close()
    return output


def plot_training_curves(results_csv: str | Path, output_dir: str | Path) -> Path:
    """Plot common Ultralytics training metrics from ``results.csv``."""

    pandas = __import__("pandas")
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    pyplot = __import__("matplotlib.pyplot", fromlist=["pyplot"])

    csv_path = resolve_path(results_csv)
    data = pandas.read_csv(csv_path)
    data.columns = [column.strip() for column in data.columns]

    metrics = [
        column for column in data.columns
        if column.startswith("metrics/") or column.startswith("val/")
    ]

    output = resolve_path(output_dir) / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    pyplot.figure(figsize=(12, 7))
    for metric in metrics:
        pyplot.plot(data["epoch"], data[metric], label=metric)
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Value")
    pyplot.title("YOLOv8 Training Curves")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(output, dpi=160)
    pyplot.close()
    return output


def sample_label_visualization(
    image_path: str | Path,
    label_path: str | Path,
    dataset_yaml: str | Path,
    output_path: str | Path,
) -> Path:
    """Draw YOLO ground-truth labels on one image."""

    cv2 = __import__("cv2")
    image = cv2.imread(str(resolve_path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    names = load_dataset_names(dataset_yaml)
    detections = []
    for class_id, x_center, y_center, box_width, box_height in parse_yolo_label(
        resolve_path(label_path)
    ):
        detections.append(
            {
                "class_id": class_id,
                "confidence": None,
                "bbox_xyxy": yolo_to_xyxy(
                    (x_center, y_center, box_width, box_height),
                    width,
                    height,
                ),
            }
        )

    output = draw_detections(image, detections, names)
    target = resolve_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), output)
    return target


def main() -> None:
    """Generate requested visualization artifacts."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("visualize")
    dataset_yaml = config["dataset"].get("custom_yaml", "data/dataset.yaml")

    if args.plot == "classes":
        output = plot_class_distribution(dataset_yaml, "runs/class_distribution.png")
        logger.info("Saved class distribution plot to %s", output)
        return

    if not args.results_csv:
        raise ValueError("--results-csv is required for --plot training")
    output = plot_training_curves(args.results_csv, "runs")
    logger.info("Saved training curves to %s", output)


if __name__ == "__main__":
    main()
