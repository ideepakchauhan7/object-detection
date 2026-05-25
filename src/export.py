"""Export trained YOLOv8 weights to deployment formats."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from utils import (
    ensure_dir,
    get_nested,
    load_config,
    require_package,
    resolve_path,
    resolve_weights,
    setup_logger,
    ultralytics_device,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line export options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", help="Model weights to export")
    parser.add_argument(
        "--formats",
        help="Comma-separated formats such as onnx,torchscript",
    )
    parser.add_argument("--imgsz", type=int, help="Export image size")
    parser.add_argument("--device", help="Device: auto, cpu, 0, 1, ...")
    parser.add_argument("--half", action="store_true", help="Use FP16 where supported")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic axes")
    return parser.parse_args()


def main() -> None:
    """Export a model to all configured formats."""

    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger("export")
    ultralytics = require_package("ultralytics")

    export_config = config.get("export", {})
    formats = (
        [item.strip() for item in args.formats.split(",") if item.strip()]
        if args.formats
        else export_config.get("formats", ["onnx", "torchscript"])
    )

    device = args.device or get_nested(config, ["runtime", "device"], "auto")
    weights = resolve_weights(args.weights, config, logger)
    model = ultralytics.YOLO(weights)
    output_dir = ensure_dir(
        get_nested(config, ["paths", "export_dir"], "models/exported")
    )
    exported_paths: list[Path] = []

    for export_format in formats:
        logger.info("Exporting %s to %s", weights, export_format)
        exported = model.export(
            format=export_format,
            imgsz=args.imgsz or export_config.get("imgsz", 640),
            device=ultralytics_device(device),
            half=args.half or export_config.get("half", False),
            dynamic=args.dynamic or export_config.get("dynamic", False),
            opset=export_config.get("opset", 12),
        )

        exported_path = resolve_path(exported)
        target = output_dir / exported_path.name
        if exported_path.exists() and exported_path.resolve() != target.resolve():
            shutil.copy2(exported_path, target)
            exported_paths.append(target)
        else:
            exported_paths.append(exported_path)

    logger.info("Exported artifacts: %s", [str(path) for path in exported_paths])


if __name__ == "__main__":
    main()
