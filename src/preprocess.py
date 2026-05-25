"""Prepare and validate YOLO-format object detection datasets.

The script creates the expected directory layout, validates YOLO ``.txt``
labels, summarizes class counts, and can optionally create offline
Albumentations copies for the training split.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from utils import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    image_files,
    load_config,
    resolve_path,
    setup_logger,
)


REQUIRED_DIRS = (
    "data/images/train",
    "data/images/val",
    "data/images/test",
    "data/labels/train",
    "data/labels/val",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preprocessing."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the dataset without augmentation.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Run offline Albumentations augmentation for the train split.",
    )
    return parser.parse_args()


def create_dataset_dirs() -> None:
    """Create the standard YOLO image and label directories."""

    for directory in REQUIRED_DIRS:
        ensure_dir(directory)


def load_dataset_names(dataset_yaml: str | Path) -> dict[int, str]:
    """Load class names from a YOLO dataset YAML file."""

    import yaml

    path = resolve_path(dataset_yaml)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    names = payload.get("names", {})
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read one YOLO label file and return normalized boxes."""

    boxes: list[tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number} must contain 5 values, "
                    f"found {len(parts)}"
                )

            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            coords = (x_center, y_center, width, height)
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise ValueError(
                    f"{label_path}:{line_number} has coordinates outside [0, 1]"
                )
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def matching_image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    """Find the image file that corresponds to a label stem."""

    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def validate_split(split: str) -> dict[str, Any]:
    """Validate one dataset split and return summary statistics."""

    image_dir = resolve_path(f"data/images/{split}")
    label_dir = resolve_path(f"data/labels/{split}")
    images = image_files(image_dir)
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    class_counts: Counter[int] = Counter()
    missing_images: list[str] = []
    missing_labels: list[str] = []
    boxes_total = 0

    for label_path in labels:
        boxes = parse_yolo_label(label_path)
        boxes_total += len(boxes)
        class_counts.update(box[0] for box in boxes)
        if matching_image_for_label(label_path, image_dir) is None:
            missing_images.append(str(label_path.relative_to(resolve_path("."))))

    label_stems = {label.stem for label in labels}
    for image_path in images:
        if image_path.stem not in label_stems and split != "test":
            missing_labels.append(str(image_path.relative_to(resolve_path("."))))

    return {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "boxes": boxes_total,
        "class_counts": dict(sorted(class_counts.items())),
        "missing_images_for_labels": missing_images,
        "missing_labels_for_images": missing_labels,
    }


def summarize_dataset(dataset_yaml: str | Path) -> dict[str, Any]:
    """Validate all available splits and summarize labels/classes."""

    names = load_dataset_names(dataset_yaml)
    splits = ["train", "val", "test"]
    split_summaries = {split: validate_split(split) for split in splits}
    return {"names": names, "splits": split_summaries}


def augment_training_data(config: dict[str, Any]) -> int:
    """Create offline augmented training images and YOLO labels.

    The augmentation is disabled by default in ``config.yaml`` because it writes
    new files into the dataset directories.
    """

    cv2 = __import__("cv2")
    albumentations = __import__("albumentations")

    copies = int(config["augmentation"]["offline"].get("copies_per_image", 1))
    image_dir = resolve_path("data/images/train")
    label_dir = resolve_path("data/labels/train")

    transform = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.4),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.Blur(blur_limit=3, p=0.15),
        ],
        bbox_params=albumentations.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.2,
        ),
    )

    created = 0
    for image_path in image_files(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        boxes = parse_yolo_label(label_path)
        if not boxes:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        class_labels = [box[0] for box in boxes]
        yolo_boxes = [box[1:] for box in boxes]

        for index in range(copies):
            augmented = transform(
                image=image,
                bboxes=yolo_boxes,
                class_labels=class_labels,
            )
            output_stem = f"{image_path.stem}_aug_{index + 1}"
            output_image = image_dir / f"{output_stem}{image_path.suffix}"
            output_label = label_dir / f"{output_stem}.txt"

            cv2.imwrite(str(output_image), augmented["image"])
            with output_label.open("w", encoding="utf-8") as handle:
                for class_id, bbox in zip(
                    augmented["class_labels"],
                    augmented["bboxes"],
                    strict=True,
                ):
                    values = " ".join(f"{value:.6f}" for value in bbox)
                    handle.write(f"{int(class_id)} {values}\n")
            created += 1

    return created


def main() -> None:
    """Run dataset directory creation, validation, and optional augmentation."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("preprocess")
    create_dataset_dirs()

    dataset_yaml = config["dataset"].get("custom_yaml", "data/dataset.yaml")
    summary = summarize_dataset(dataset_yaml)
    logger.info("Dataset summary: %s", summary)

    should_augment = args.augment or (
        config.get("augmentation", {}).get("offline", {}).get("enabled", False)
        and not args.validate_only
    )
    if should_augment:
        created = augment_training_data(config)
        logger.info("Created %s augmented training images.", created)


if __name__ == "__main__":
    main()
