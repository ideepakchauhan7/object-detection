"""Shared utilities for the YOLOv8 object detection project.

This module keeps path handling, configuration loading, logging, and device
selection in one place so the training, evaluation, inference, and export
scripts behave consistently.
"""

from __future__ import annotations

import importlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def require_package(module_name: str, pip_name: str | None = None) -> Any:
    """Import an optional dependency with a clear installation message.

    Args:
        module_name: Python import name, for example ``ultralytics``.
        pip_name: Package name to show in the error if it differs.

    Returns:
        The imported module.

    Raises:
        RuntimeError: If the dependency is missing.
    """

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = pip_name or module_name
        raise RuntimeError(
            f"Missing dependency '{package}'. Install project dependencies with "
            "'pip install -r requirements.txt'."
        ) from exc


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the central YAML project configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary with project metadata added.
    """

    yaml = require_package("yaml", "PyYAML")
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(path_value: str | Path, base: Path | None = None) -> Path:
    """Resolve a path relative to the project root by default."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT).joinpath(path).resolve()


def ensure_dir(path_value: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""

    path = resolve_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_nested(config: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Read a nested config value without repetitive ``dict.get`` chains."""

    cursor: Any = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def setup_logger(
    name: str = "yolo_project",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console/file logger without duplicating handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not any(isinstance(handler, logging.StreamHandler)
               for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        log_path = resolve_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_device(device_config: str | int | None = "auto") -> str:
    """Return ``cuda:0`` when available, otherwise ``cpu``.

    Args:
        device_config: ``auto``, ``cpu``, CUDA index, or a device string.

    Returns:
        Device string for general PyTorch-style use.
    """

    if device_config is None or str(device_config).lower() == "auto":
        try:
            torch = importlib.import_module("torch")
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return str(device_config)


def ultralytics_device(device_config: str | int | None = "auto") -> str:
    """Normalize device values for the Ultralytics Python API."""

    device = get_device(device_config)
    if device.startswith("cuda"):
        # Ultralytics accepts GPU indexes such as "0".
        return device.split(":", maxsplit=1)[1] if ":" in device else "0"
    return device


def dataset_yaml_from_config(
    config: dict[str, Any],
    override: str | None = None,
) -> str:
    """Select the dataset YAML for demo or custom training."""

    if override:
        override_path = Path(override)
        looks_like_path = (
            override_path.is_absolute()
            or "/" in override
            or override.startswith(".")
        )
        if looks_like_path or resolve_path(override).exists():
            return str(resolve_path(override))
        return override

    active = get_nested(config, ["dataset", "active"], "coco128")
    if active == "custom":
        custom_yaml = get_nested(
            config,
            ["dataset", "custom_yaml"],
            "data/dataset.yaml",
        )
        return str(resolve_path(custom_yaml))
    return get_nested(config, ["dataset", "demo_yaml"], "coco128.yaml")


def resolve_weights(
    weights: str | Path | None,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> str:
    """Return existing trained weights or fall back to pretrained weights."""

    candidate = weights or get_nested(config, ["model", "trained_weights"])
    if candidate:
        candidate_path = resolve_path(candidate)
        if candidate_path.exists():
            return str(candidate_path)
        if weights and "/" not in str(weights) and Path(str(weights)).suffix == ".pt":
            return str(weights)

    fallback = get_nested(config, ["model", "pretrained"], "yolov8n.pt")
    if logger:
        logger.warning(
            "Trained weights were not found; using pretrained weights: %s",
            fallback,
        )
    return str(fallback)


def copy_training_weights(save_dir: str | Path, models_dir: str | Path) -> list[Path]:
    """Copy Ultralytics ``best.pt`` and ``last.pt`` into the models folder."""

    source_dir = resolve_path(save_dir)
    destination = ensure_dir(models_dir)
    copied: list[Path] = []

    for name in ("best.pt", "last.pt"):
        source = source_dir / "weights" / name
        if source.exists():
            target = destination / name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write a JSON artifact with stable formatting."""

    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def image_files(directory: str | Path) -> list[Path]:
    """Return all image files in a directory recursively."""

    root = resolve_path(directory)
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_source(source: str) -> str | int:
    """Convert webcam source indexes such as ``0`` to integers."""

    return int(source) if str(source).isdigit() else source
