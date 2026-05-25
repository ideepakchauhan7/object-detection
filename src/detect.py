"""Run YOLOv8 inference on images, directories, videos, or webcam streams."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from utils import (
    get_nested,
    load_config,
    parse_source,
    require_package,
    resolve_path,
    resolve_weights,
    setup_logger,
    ultralytics_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line inference options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", help="Model weights for inference")
    parser.add_argument(
        "--source",
        help="Image, directory, video, URL, or webcam index",
    )
    parser.add_argument("--conf", type=float, help="Confidence threshold")
    parser.add_argument("--iou", type=float, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", help="Device: auto, cpu, 0, 1, ...")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt results")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped objects")
    parser.add_argument("--show", action="store_true", help="Display output windows")
    return parser.parse_args()


def detections_from_results(results: list[Any]) -> list[dict[str, Any]]:
    """Convert Ultralytics predictions to serializable detection rows."""

    rows: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {})
        path = str(getattr(result, "path", ""))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            rows.append(
                {
                    "source": path,
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return rows


def write_detections_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write detection rows to ``detections.csv``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detections.csv"
    fieldnames = [
        "source",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    """Run inference and save annotated outputs plus a detection CSV."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("detect")
    ultralytics = require_package("ultralytics")

    detection = config.get("detection", {})
    device = args.device or get_nested(config, ["runtime", "device"], "auto")
    weights = resolve_weights(args.weights, config, logger)
    source = args.source or detection.get("source", "data/images/test")

    model = ultralytics.YOLO(weights)
    results = model.predict(
        source=parse_source(str(source)),
        conf=args.conf if args.conf is not None else detection.get("conf", 0.25),
        iou=args.iou if args.iou is not None else detection.get("iou", 0.45),
        imgsz=args.imgsz or detection.get("imgsz", 640),
        device=ultralytics_device(device),
        project=str(
            resolve_path(
                get_nested(config, ["paths", "detection_project"], "runs/detect")
            )
        ),
        name=detection.get("name", "predict"),
        save=True,
        save_txt=args.save_txt or detection.get("save_txt", False),
        save_crop=args.save_crop or detection.get("save_crop", False),
        show=args.show or detection.get("show", False),
    )

    rows = detections_from_results(results)
    if results:
        save_dir = Path(getattr(results[0], "save_dir", resolve_path("runs/detect")))
    else:
        save_dir = resolve_path(
            f"{get_nested(config, ['paths', 'detection_project'], 'runs/detect')}/"
            f"{detection.get('name', 'predict')}"
        )
    csv_path = write_detections_csv(rows, save_dir)
    logger.info("Saved %s detections to %s", len(rows), csv_path)


if __name__ == "__main__":
    main()
